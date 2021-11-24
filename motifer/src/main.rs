use motifer;
use std::env;
use std::time;

//On Jeremy's laptop, MI calculation took 35.85545999308333 minutes 
//  I ran target/release/motifer ../test_data/shapes.npy ../test_data/y_vals.npy ../test_data/test_args.pkl

fn main() {
    let args: Vec<String> = env::args().collect();
    let cfg = motifer::parse_config(&args);
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
