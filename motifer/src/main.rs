use motifer;
use std::env;
use std::time;
use rayon::ThreadPoolBuilder;

//  I ran target/release/motifer ../test_data/shapes.npy ../test_data/y_vals.npy ../test_data/config.pkl ../test_data/test_output.pkl
// On Jeremy's laptop, run in series:
//   MI calculation took 36.28 minutes 
// Running on lighthouse to see how that does, run in series:
//   MI calculation took 38.79 minutes
// Python's MI calculation on 1 core on lighthouse:
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

    let threshold = motifer::set_initial_threshold(
        &seeds,
        &rec_db,
        cfg.seed_sample_size,
        cfg.records_per_seed,
        cfg.windows_per_record,
        &cfg.kmer,
        &cfg.alpha,
        cfg.thresh_sd_from_mean,
    );

    let now = time::Instant::now();
    seeds.compute_mi_values(
        &rec_db,
        &threshold,
        &cfg.max_count,
    );
    let duration = now.elapsed().as_secs_f64() / 60.0;
    println!("MI calculation took {:?} minutes.", duration);

    println!("{} seeds prior to CMI-based filtering.", seeds.len());
    let motifs = motifer::filter_seeds(
        &mut seeds,
        &rec_db,
        &threshold,
        &cfg.max_count,
    );
    println!("{} motifs left after CMI-based filtering.", motifs.len());

    motifer::pickle_motifs(
        &motifs,
        &cfg.out_fname,
    );
    println!("Vector of motifs written to: {}", &cfg.out_fname);
}
