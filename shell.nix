let
  pkgs = import <nixpkgs> {};
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    pkg-config
    python312
    python312Packages.pip
    python312Packages.numpy
    python312Packages.matplotlib
    python312Packages.fastapi
    python312Packages.librosa
    python312Packages.jupyterlab
    python312Packages.pandas
    python312Packages.pyarrow
    python312Packages.fastparquet
    python312Packages.torch
    python312Packages.torchvision
    python312Packages.torchaudio
    python312Packages.scikit-learn
    python312Packages.seaborn
  ];

  shellHook = ''
    echo "env is ready"
  '';
}
