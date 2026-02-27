{
  description = "Physique 2";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  outputs =
    { self, nixpkgs }:
    {
      devShells.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.mkShell {
        name = "data-processing";
        packages = with nixpkgs.legacyPackages.x86_64-linux; [
          julia_111-bin
        ];
        shellHook = ''
          clear
          fastfetch
          echo -e "\n"
          echo "Hello $USER, environnement physique 2 => activated ✅"

          julia -e '
            using Pkg
            if !haskey(Pkg.project().dependencies, "Pluto")
              println("Installation de Pluto...")
              Pkg.add("Pluto")
            else
              println("Pluto déjà installé")
            end
          '

          alias pluto="julia -e 'using Pluto; Pluto.run()'"

          echo ""
          echo "Lance Pluto avec : pluto"
        '';
      };
    };
}
