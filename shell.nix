let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  buildInputs = [
    pkgs.python38
    pkgs.python38Packages.pip
    pkgs.python38Packages.pysdl2
    pkgs.python38Packages.pillow
    pkgs.python38Packages.pytorch
    pkgs.python38Packages.torchvision
    pkgs.python38Packages.numpy
  ];
  shellHook = ''
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python38.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib"
    unset SOURCE_DATE_EPOCH
  '';
}
