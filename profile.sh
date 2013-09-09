if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
  . "$HOME/.bashrc"
    fi
fi

# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/bin" ] ; then
    PATH="$HOME/bin:$PATH"
fi

if [ -d /usr/local/cuda ]; then
	PATH="$PATH:/usr/local/cuda/bin"
	LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
fi

export PATH LD_LIBRARY_PATH
