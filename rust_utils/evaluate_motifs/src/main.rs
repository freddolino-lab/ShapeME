use motifer;
use std::path;
use std::env;
use rayon::ThreadPoolBuilder;

fn main() {
    let args: Vec<String> = env::args().collect();
    let cfg = motifer::parse_config(&args);

    ThreadPoolBuilder::new()
        .num_threads(cfg.cores)
        .build_global()
        .unwrap();

    let rec_db = motifer::RecordsDB::new_from_files(
        &cfg.eval_shape_fname,
        &cfg.eval_yvals_fname,
    );

    // read motifs
    println!("Reading motifs from {:?}", &cfg.out_fname);
    let mut motifs = motifer::read_motifs(&cfg.out_fname);;

    motifs.post_optim_update(&rec_db, &cfg.max_count);

    let parent = path::Path::new(&cfg.out_fname).parent().unwrap();

    let eval_motifs_fname = parent.join("evaluated_motifs.json");
    println!("Rust binary writing evaluated motifs to {:?}", &eval_motifs_fname);
    motifs.json_motifs(&eval_motifs_fname.to_str().unwrap());
}

