from __future__ import annotations

import unittest

import numpy as np
import torch

from object_physics_latent.encoder import (
    ObjectPhysicsEncoder,
    VisualPointSetEncoder,
    latent_regularization_losses,
    same_object_consistency_loss,
    symmetric_info_nce_loss,
)
from object_physics_latent.friction_decoder import (
    LatentConditionedFrictionDecoder,
    build_point_conditioning_features,
)
from object_physics_latent.model import TrajectoryConditionedFrictionModel


class ObjectPhysicsModelTest(unittest.TestCase):
    def test_encoder_shapes_masks_and_losses(self) -> None:
        torch.manual_seed(0)
        batch_size = 3
        trajectories = 4
        steps = 11
        feature_dim = 12
        features = torch.randn(batch_size, trajectories, steps, feature_dim)
        valid_mask = torch.ones(batch_size, trajectories, steps, dtype=torch.bool)
        valid_mask[0, 0, 7:] = False
        valid_mask[1, 2, 5:] = False

        encoder = ObjectPhysicsEncoder(
            input_dim=feature_dim,
            latent_dim=8,
            projection_dim=16,
            step_hidden_dim=32,
            gru_hidden_dim=32,
            trajectory_embedding_dim=32,
            set_hidden_dim=32,
            projection_hidden_dim=16,
        )
        output_a = encoder(features, valid_mask=valid_mask)
        output_b = encoder(features + 0.01 * torch.randn_like(features), valid_mask=valid_mask)

        self.assertEqual(tuple(output_a.latent.shape), (batch_size, 8))
        self.assertEqual(tuple(output_a.projection.shape), (batch_size, 8))
        self.assertEqual(tuple(output_a.trajectory_embeddings.shape), (batch_size, trajectories, 32))
        self.assertTrue(torch.allclose(torch.linalg.norm(output_a.latent, dim=-1), torch.ones(batch_size), atol=1e-5))
        self.assertTrue(torch.allclose(output_a.latent, output_a.projection, atol=1e-6))

        consistency = same_object_consistency_loss(output_a.latent, output_b.latent)
        contrastive = symmetric_info_nce_loss(output_a.latent, output_b.latent, temperature=0.1)
        regularization = latent_regularization_losses(output_a, output_b)
        self.assertTrue(torch.isfinite(consistency))
        self.assertTrue(torch.isfinite(contrastive))
        self.assertTrue(torch.isfinite(regularization.total))
        self.assertTrue(torch.isfinite(regularization.latent_norm))

        regularization.total.backward()
        grad_norm = sum(
            float(param.grad.detach().abs().sum())
            for param in encoder.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_encoder_visual_features_fuse_into_latent(self) -> None:
        torch.manual_seed(7)
        encoder = ObjectPhysicsEncoder(
            input_dim=12,
            latent_dim=8,
            projection_dim=16,
            step_hidden_dim=32,
            gru_hidden_dim=32,
            trajectory_embedding_dim=32,
            set_hidden_dim=32,
            visual_input_dim=17,
            visual_hidden_dim=32,
            visual_embedding_dim=24,
            projection_hidden_dim=16,
        )
        context = torch.randn(2, 3, 9, 12)
        valid_mask = torch.ones(2, 3, 9, dtype=torch.bool)
        visual_a = torch.randn(2, 5, 17)
        visual_b = visual_a.clone()
        visual_b[1] = visual_b[1] + 0.5

        output_a = encoder(context, valid_mask=valid_mask, visual_features=visual_a)
        output_b = encoder(context, valid_mask=valid_mask, visual_features=visual_b)

        self.assertEqual(tuple(output_a.latent.shape), (2, 8))
        self.assertEqual(tuple(output_a.projection.shape), (2, 8))
        self.assertTrue(torch.allclose(torch.linalg.norm(output_a.latent, dim=-1), torch.ones(2), atol=1e-5))
        self.assertEqual(tuple(output_a.visual_embedding.shape), (2, 24))
        self.assertGreater(float(torch.abs(output_a.latent[1] - output_b.latent[1]).sum().detach()), 0.0)

        loss = output_a.latent.mean()
        loss.backward()
        visual_grad = sum(
            float(param.grad.detach().abs().sum())
            for param in encoder.visual_encoder.parameters()
            if param.grad is not None
        )
        self.assertGreater(visual_grad, 0.0)

    def test_visual_point_set_encoder_shapes(self) -> None:
        encoder = VisualPointSetEncoder(input_dim=6, hidden_dim=8, embedding_dim=5)
        point_features = torch.randn(3, 7, 6)
        output = encoder(point_features)
        self.assertEqual(tuple(output.shape), (3, 5))
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_friction_decoder_bounds_active_indices_and_gradients(self) -> None:
        torch.manual_seed(1)
        decoder = LatentConditionedFrictionDecoder(
            point_feature_dim=10,
            latent_dim=8,
            hidden_dim=32,
            hidden_layers=2,
            mu_min=0.1,
            mu_max=0.7,
            initial_mu=0.35,
        )
        point_features = torch.randn(12, 10, requires_grad=True)
        latent = torch.randn(2, 8, requires_grad=True)
        active_indices = torch.tensor([0, 2, 5, 7, 11], dtype=torch.long)

        mu = decoder(point_features, latent, active_indices=active_indices)
        self.assertEqual(tuple(mu.shape), (2, 5))
        self.assertGreaterEqual(float(mu.min().detach()), 0.1)
        self.assertLessEqual(float(mu.max().detach()), 0.7)

        loss = mu.mean()
        loss.backward()
        self.assertIsNotNone(point_features.grad)
        self.assertIsNotNone(latent.grad)
        self.assertGreater(float(point_features.grad.detach().abs().sum()), 0.0)
        self.assertGreater(float(latent.grad.detach().abs().sum()), 0.0)

    def test_friction_decoder_negative_hidden_activations_keep_gradients(self) -> None:
        decoder = LatentConditionedFrictionDecoder(
            point_feature_dim=3,
            latent_dim=2,
            hidden_dim=4,
            hidden_layers=2,
            mu_min=0.0,
            mu_max=1.0,
            initial_mu=0.35,
        )
        self.assertEqual(decoder.conditioning, "film")
        self.assertIsInstance(decoder.activation, torch.nn.SiLU)
        self.assertFalse(hasattr(decoder, "latent_output"))
        with torch.no_grad():
            for layer in decoder.point_layers:
                layer.weight.fill_(0.1)
                layer.bias.fill_(-1.0)
            for layer in decoder.film_layers:
                layer.weight.fill_(0.1)
                layer.bias.zero_()
            decoder.output_layer.weight.fill_(0.1)
            decoder.output_layer.bias.fill_(-1.0)

        point_features = torch.zeros(5, 3, requires_grad=True)
        latent = torch.zeros(1, 2, requires_grad=True)
        mu = decoder(point_features, latent)

        hidden = point_features.unsqueeze(0)
        for point_layer, film_layer in zip(decoder.point_layers, decoder.film_layers, strict=True):
            hidden = point_layer(hidden)
            gamma, beta = film_layer(latent).chunk(2, dim=-1)
            hidden = decoder.activation(hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1))
        self.assertTrue(torch.all(hidden < 0.0))

        mu.sum().backward()
        self.assertGreater(float(point_features.grad.detach().abs().sum()), 0.0)
        self.assertGreater(float(latent.grad.detach().abs().sum()), 0.0)

    def test_concat_decoder_mode_remains_available_for_legacy_checkpoints(self) -> None:
        decoder = LatentConditionedFrictionDecoder(
            point_feature_dim=3,
            latent_dim=2,
            hidden_dim=4,
            hidden_layers=2,
            conditioning="concat",
            activation="relu",
        )
        self.assertTrue(any(isinstance(layer, torch.nn.ReLU) for layer in decoder.mlp))
        mu = decoder(torch.zeros(5, 3), torch.zeros(1, 2))
        self.assertEqual(tuple(mu.shape), (1, 5))
        self.assertTrue(torch.all(torch.isfinite(mu)))

    def test_basis_decoder_uses_latent_coefficients_and_point_basis(self) -> None:
        torch.manual_seed(11)
        decoder = LatentConditionedFrictionDecoder(
            point_feature_dim=6,
            latent_dim=4,
            hidden_dim=16,
            hidden_layers=2,
            conditioning="basis",
            activation="silu",
            basis_count=3,
            mu_min=0.0,
            mu_max=1.0,
            initial_mu=0.35,
        )
        self.assertEqual(decoder.conditioning, "basis")
        self.assertEqual(decoder.basis_count, 3)
        self.assertTrue(hasattr(decoder, "basis_net"))
        self.assertTrue(hasattr(decoder, "coef_head"))
        self.assertTrue(hasattr(decoder, "base_head"))

        point_features = torch.randn(9, 6, requires_grad=True)
        latent = torch.randn(2, 4, requires_grad=True)
        active_indices = torch.tensor([0, 1, 4, 8], dtype=torch.long)
        mu = decoder(point_features, latent, active_indices=active_indices)

        self.assertEqual(tuple(mu.shape), (2, 4))
        self.assertGreaterEqual(float(mu.min().detach()), 0.0)
        self.assertLessEqual(float(mu.max().detach()), 1.0)

        weights = torch.tensor([0.2, 0.7, -0.3, 1.1], dtype=mu.dtype)
        loss = (mu * weights.unsqueeze(0)).sum()
        loss.backward()
        self.assertIsNotNone(point_features.grad)
        self.assertIsNotNone(latent.grad)
        self.assertGreater(float(point_features.grad.detach().abs().sum()), 0.0)
        self.assertGreater(float(latent.grad.detach().abs().sum()), 0.0)

    def test_basis_decoder_global_shared_base_and_unit_std_basis(self) -> None:
        torch.manual_seed(13)
        decoder = LatentConditionedFrictionDecoder(
            point_feature_dim=5,
            latent_dim=3,
            hidden_dim=12,
            hidden_layers=2,
            conditioning="basis",
            activation="silu",
            basis_count=4,
            basis_base_mode="global_shared",
            basis_normalization="unit_std",
            basis_activation="identity",
            mu_min=0.0,
            mu_max=1.0,
            initial_mu=0.35,
        )
        self.assertFalse(hasattr(decoder, "base_head"))
        self.assertTrue(hasattr(decoder, "basis_base"))
        self.assertTrue(isinstance(decoder.basis_base, torch.nn.Parameter))

        point_features = torch.randn(11, 5, requires_grad=True)
        latent = torch.randn(2, 3, requires_grad=True)
        mu = decoder(point_features, latent)
        diagnostics = decoder.basis_diagnostics(point_features, latent)

        self.assertEqual(tuple(mu.shape), (2, 11))
        self.assertGreater(float(diagnostics["basis_std"]), 0.5)
        self.assertGreater(float(diagnostics["spatial_raw_std"]), 0.0)
        weights = torch.linspace(-0.5, 0.5, steps=11, dtype=mu.dtype)
        loss = (mu * weights.unsqueeze(0)).sum()
        loss.backward()
        self.assertIsNotNone(decoder.basis_base.grad)
        self.assertIsNotNone(point_features.grad)
        self.assertIsNotNone(latent.grad)
        self.assertGreater(float(point_features.grad.detach().abs().sum()), 0.0)
        self.assertGreater(float(latent.grad.detach().abs().sum()), 0.0)

    def test_basis_decoder_fixed_base_has_no_trainable_base(self) -> None:
        decoder = LatentConditionedFrictionDecoder(
            point_feature_dim=4,
            latent_dim=3,
            hidden_dim=10,
            hidden_layers=1,
            conditioning="basis",
            basis_count=2,
            basis_base_mode="fixed",
            basis_normalization="unit_std",
            basis_activation="identity",
            mu_min=0.0,
            mu_max=1.0,
            initial_mu=0.35,
        )
        self.assertFalse(hasattr(decoder, "base_head"))
        self.assertNotIn("basis_base", dict(decoder.named_parameters()))
        self.assertIn("basis_base", dict(decoder.named_buffers()))

    def test_decoder_latent_normalization_is_compat_only_and_scale_can_change_output(self) -> None:
        decoder = LatentConditionedFrictionDecoder(
            point_feature_dim=3,
            latent_dim=4,
            hidden_dim=8,
            hidden_layers=1,
            conditioning="concat",
            activation="silu",
            latent_normalization="layernorm",
            raw_limit=1.0,
            mu_min=0.0,
            mu_max=2.0,
            initial_mu=0.35,
        )
        self.assertIsNone(decoder.latent_norm_layer)
        point_features = torch.randn(6, 3)
        latent_small = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
        latent_large = latent_small * 100.0
        mu_small = decoder(point_features, latent_small)
        mu_large = decoder(point_features, latent_large)
        self.assertFalse(torch.allclose(mu_small, mu_large, atol=1.0e-5))
        self.assertGreaterEqual(
            float(mu_small.detach().min()),
            2.0 * torch.sigmoid(torch.tensor(-1.0)).item() - 1.0e-6,
        )
        self.assertLessEqual(
            float(mu_small.detach().max()),
            2.0 * torch.sigmoid(torch.tensor(1.0)).item() + 1.0e-6,
        )

    def test_latent_norm_regularization_tracks_unit_latents(self) -> None:
        torch.manual_seed(13)
        encoder = ObjectPhysicsEncoder(
            input_dim=12,
            latent_dim=8,
            projection_dim=16,
            step_hidden_dim=32,
            gru_hidden_dim=32,
            trajectory_embedding_dim=32,
            set_hidden_dim=32,
        )
        features = torch.randn(2, 3, 9, 12)
        output_a = encoder(features)
        output_b = encoder(features + 0.01 * torch.randn_like(features))
        regularization = latent_regularization_losses(
            output_a,
            output_b,
        )
        self.assertLess(float(regularization.latent_norm.detach()), 1.0e-5)
        regularization.total.backward()
        grad_norm = sum(
            float(param.grad.detach().abs().sum())
            for param in encoder.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_position_only_point_feature_builder(self) -> None:
        points = np.array(
            [
                [-0.1, -0.05, -0.025],
                [0.0, 0.0, -0.025],
                [0.1, 0.05, -0.025],
            ],
            dtype=np.float32,
        )
        features, metadata, stats = build_point_conditioning_features(
            local_surface_points=points,
            half_extents=np.array([0.1, 0.05, 0.025], dtype=np.float32),
            dino_features=None,
            position_frequencies=2,
        )
        self.assertEqual(features.shape[0], 3)
        self.assertEqual(features.shape[1], metadata.input_dim)
        self.assertEqual(metadata.dino_dim, 0)
        self.assertIn("encoded_position_dim", stats)

    def test_combined_model_forward(self) -> None:
        torch.manual_seed(2)
        model = TrajectoryConditionedFrictionModel.from_dimensions(
            point_feature_dim=9,
            visual_feature_dim=11,
            visual_hidden_dim=32,
            visual_embedding_dim=16,
            encoder_feature_dim=12,
            latent_dim=8,
            projection_dim=16,
            step_hidden_dim=32,
            gru_hidden_dim=32,
            trajectory_embedding_dim=32,
            set_hidden_dim=32,
            decoder_hidden_dim=32,
            decoder_hidden_layers=1,
            mu_min=0.0,
            mu_max=1.0,
            initial_mu=0.35,
        )
        context_features = torch.randn(2, 4, 10, 12)
        context_mask = torch.ones(2, 4, 10, dtype=torch.bool)
        point_features = torch.randn(20, 9)
        visual_features = torch.randn(2, 20, 11)
        output = model(
            context_features=context_features,
            context_valid_mask=context_mask,
            point_features=point_features,
            visual_features=visual_features,
        )
        self.assertEqual(tuple(output.latent.shape), (2, 8))
        self.assertEqual(tuple(output.projection.shape), (2, 8))
        self.assertTrue(torch.allclose(torch.linalg.norm(output.latent, dim=-1), torch.ones(2), atol=1e-5))
        self.assertEqual(tuple(output.friction.shape), (2, 20))
        self.assertTrue(torch.all(torch.isfinite(output.friction)))

    def test_combined_model_basis_forward(self) -> None:
        torch.manual_seed(12)
        model = TrajectoryConditionedFrictionModel.from_dimensions(
            point_feature_dim=13,
            visual_feature_dim=13,
            encoder_feature_dim=12,
            latent_dim=6,
            projection_dim=10,
            step_hidden_dim=24,
            gru_hidden_dim=24,
            trajectory_embedding_dim=24,
            set_hidden_dim=24,
            visual_hidden_dim=16,
            visual_embedding_dim=16,
            decoder_hidden_dim=16,
            decoder_hidden_layers=1,
            decoder_conditioning="basis",
            decoder_basis_count=4,
            mu_min=0.0,
            mu_max=1.0,
            initial_mu=0.35,
        )
        context_features = torch.randn(3, 2, 8, 12)
        point_features = torch.randn(3, 15, 13)
        visual_features = torch.randn(3, 15, 13)
        output = model(
            context_features=context_features,
            point_features=point_features,
            visual_features=visual_features,
        )
        self.assertEqual(tuple(output.latent.shape), (3, 6))
        self.assertEqual(tuple(output.friction.shape), (3, 15))
        self.assertEqual(model.friction_decoder.basis_count, 4)
        self.assertEqual(tuple(output.projection.shape), (3, 6))
        self.assertTrue(torch.allclose(torch.linalg.norm(output.latent, dim=-1), torch.ones(3), atol=1e-5))
        self.assertTrue(torch.all(torch.isfinite(output.friction)))


if __name__ == "__main__":
    unittest.main()
