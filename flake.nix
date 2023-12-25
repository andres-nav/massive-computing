{
  description = "Shells for uni";

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
  in {
    devShells.${system}.default = with import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    }; let
      custom-python = python310.withPackages (p:
        with p; [
          ipykernel
          pandas
          pyspark
          black
          pycuda
          pyarrow
        ]);
    in
      mkShell {
        packages = [
          zulu
          custom-python
          cudaPackages.cuda_nvcc
          cudaPackages.cuda_cccl
          cudaPackages.cudatoolkit
        ];
      };
  };
}
