if not contains "/workspace/.local/bin" $PATH
    # Prepending path in case a system-installed binary needs to be overridden
    set -x PATH "/workspace/.local/bin" $PATH
end
