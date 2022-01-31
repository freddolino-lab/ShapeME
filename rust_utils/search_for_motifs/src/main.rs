use motifer;
use optim;
use ndarray_npy;
use std::path;
use std::env;
use std::time;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

// Run target/release/find_motifs ../test_data/shapes.npy ../test_data/y_vals.npy ../test_data/config.pkl ../test_data/test_output.pkl

fn main() {
    let args: Vec<String> = env::args().collect();
    let cfg = motifer::parse_config(&args);

    ThreadPoolBuilder::new()
        .num_threads(cfg.cores)
        .build_global()
        .unwrap();

    let rec_db = motifer::RecordsDB::new_from_files(
        &cfg.shape_fname,
        &cfg.yvals_fname,
    );

    // read motifs
    let mut motifs = motifer::read_motifs(&cfg.out_fname);;

    motifs.post_optim_update(&rec_db, &cfg.max_count);

    let parent = path::Path::new(&cfg.out_fname).parent().unwrap();

    let eval_motifs_fname = parent.join("evaluated_motifs.json");
    motifs.json_motifs(&eval_motifs_fname);
}

