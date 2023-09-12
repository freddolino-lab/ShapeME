use motifer;
use std::path;
use std::env;
use rayon::ThreadPoolBuilder;

fn main() {
    // first argument must go config file
    let args: Vec<String> = env::args().collect();
    let cfg = motifer::parse_config(&args).unwrap();

    ThreadPoolBuilder::new()
        .num_threads(cfg.cores)
        .build_global()
        .unwrap();

    println!("shape fname: {}", &cfg.shape_fname);
    println!("yval fname: {}", &cfg.yvals_fname);
    let rec_db = motifer::RecordsDB::new_from_files(
        &cfg.shape_fname,
        &cfg.yvals_fname,
    );

    // read motifs
    println!("Reading motifs from {:?}", &cfg.eval_rust_fname);
    let mut motifs = motifer::read_motifs(&cfg.eval_rust_fname);

    motifs.fold_merge_update(&rec_db, &cfg.max_count, &cfg.alpha);
    motifs.json_motifs("test_motif_result.json");

    println!(
        "{} shape motifs from all folds. Attempting CMI-based filtering of motifs.",
        motifs.len(),
    );
    let motifs = motifs.filter_motifs(
        &rec_db,
        &cfg.max_count,
    );
    println!("{} motifs left after pooled CMI-based filtering.", motifs.len());

    //let parent = path::Path::new(&cfg.eval_rust_fname).parent().unwrap();

    //let merged_motifs_fname = parent.join("merged_shape_motifs.json");
    println!("Writing merged motifs to {:?}", &cfg.out_fname);
    motifs.json_motifs(&cfg.out_fname);
}

