use motifer;
use std::path;
use std::env;
use rayon::ThreadPoolBuilder;

fn main() {
    let args: Vec<String> = env::args().collect();
    let cfg = motifer::parse_config(&args).unwrap();

    ThreadPoolBuilder::new()
        .num_threads(cfg.cores)
        .build_global()
        .unwrap();

    let rec_db = motifer::RecordsDB::new_from_files(
        &cfg.eval_shape_fname,
        &cfg.eval_yvals_fname,
    );

    // read motifs
    println!("Reading motifs from {:?}", &cfg.eval_rust_fname);
    let mut motifs = motifer::read_motifs(&cfg.eval_rust_fname);
    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    // I need to filter by zscore or ami, or something here.
    ///////////////////////////////////////////////////////////////////

    motifs.filter_by_zscore();
    motifs.post_optim_update(&rec_db, &cfg.max_count);

    // so at this point I need to write a table that looks just like fimo output
    ////////////////////////////////////////////////////////////////////////
    let parent = path::Path::new(&cfg.out_fname).parent().unwrap();
    let fimo_fname = parent.join("locate_motifs_results.tsv");
    motifs.write_fimo();

    //println!("locate_motifs writing motifs to {:?}", &eval_motifs_fname);
    //motifs.json_motifs(&eval_motifs_fname.to_str().unwrap());
}

