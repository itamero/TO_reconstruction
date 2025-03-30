import argparse
from ab_initio_TO import cryo_abinitio_TO

def main():
    parser = argparse.ArgumentParser(description="Run cryo_abinitio_TO with specified parameters.")

    parser.add_argument("--sym", type=str, help="Symmetry type of the molecule")
    parser.add_argument("--instack", type=str, default=None, help="Input stack file")
    parser.add_argument("--outvol", type=str, default=None, help="Output volume file")
    parser.add_argument("--cache_file_name", type=str, default=None, help="Cache file name")
    parser.add_argument("--n_theta", type=int, default=360, help="Radial resolution")
    parser.add_argument("--rotation_resolution", type=int, default=150, help="Rotation resolution")
    parser.add_argument("--n_r_perc", type=float, default=50, help="Radial percentage")
    parser.add_argument("--viewing_direction", type=float, default=0.996, help="Viewing direction threshold")
    parser.add_argument("--in_plane_rotation", type=float, default=5, help="In-plane rotation threshold")
    #parser.add_argument("--shift_step", type=float, default=0.5, help="Shift step size")
    #parser.add_argument("--max_shift_perc", type=float, default=0, help="Maximum shift percentage")
    parser.add_argument("--cg_max_iterations", type=int, default=50, help="Max iterations for conjugate gradient")

    args = parser.parse_args()

    cryo_abinitio_TO(
        sym=args.sym,
        instack=args.instack,
        outvol=args.outvol,
        cache_file_name=args.cache_file_name,
        n_theta=args.n_theta,
        rotation_resolution=args.rotation_resolution,
        n_r_perc=args.n_r_perc,
        viewing_direction=args.viewing_direction,
        in_plane_rotation=args.in_plane_rotation,
        #shift_step=args.shift_step,
        #max_shift_perc=args.max_shift_perc,
        cg_max_iterations=args.cg_max_iterations
    )

if __name__ == "__main__":
    main()
