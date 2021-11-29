use motifer;
use std::env;
use std::time;
use rayon::ThreadPoolBuilder;

//  I ran target/release/motifer ../test_data/shapes.npy ../test_data/y_vals.npy ../test_data/test_args.pkl
// On Jeremy's laptop
//   MI calculation took 36.28 minutes 
// Running on lighthouse to see how that does
//   MI calculation took 38.79 minutes
// Python's MI calculation on 1 core on lighthouse
//   MI calculation took 409.50 minutes

fn main() {
    let args: Vec<String> = env::args().collect();
    let cfg = motifer::parse_config(&args);

    ThreadPoolBuilder::new().num_threads(cfg.cores).build_global().unwrap();

    let rec_db = motifer::RecordsDB::new_from_files(
        cfg.shape_fname,
        cfg.yvals_fname,
    );
    let mut seeds = rec_db.make_seed_vec(cfg.kmer, cfg.alpha);
    let now = time::Instant::now();
    seeds.compute_mi_values(
        &rec_db,
        cfg.threshold,
        cfg.max_count,
    );
    let duration = now.elapsed().as_secs_f64() / 60.0;
    println!("MI calculation took {:?} minutes", duration)
}
