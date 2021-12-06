use motifer;
use optim;
use std::env;
use std::time;
use std::collections::HashMap;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

//  I ran target/release/find_motifs ../test_data/shapes.npy ../test_data/y_vals.npy ../test_data/config.pkl ../test_data/test_output.pkl
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

    let shape_lb = -4.0;
    let shape_ub = 4.0;
    let weights_lb = -4.0;
    let weights_ub = 4.0;
    let thresh_lb = 0.0;
    let thresh_ub = 5.0;

    let mut optimized_motifs = Vec::new();
    for motif in motifs.iter() {
        let (params,low,up) = motifer::wrangle_params_for_optim(
            &motif,
            &shape_lb,
            &shape_ub,
            &weights_lb,
            &weights_ub,
            &thresh_lb,
            &thresh_ub,
        );

        let temp = 1.0;
        let step = 0.25;
        let params_copy = params.to_vec();
        
        let mut particle = optim::Particle::new(
            params_copy,
            low,
            up,
            temp,
            step,
            &motifer::optim_objective,
            &rec_db,
            &cfg.kmer,
            &cfg.max_count,
            &cfg.alpha,
        );

        let n_iter = 1000;
        let t_adjust = 0.05;
        
        let optimized_result = optim::simulated_annealing(
            &mut particle,
            n_iter,
            &t_adjust,
            &rec_db,
            &cfg.kmer,
            &cfg.max_count,
            &cfg.alpha,
        );
        let optimized_motif = motifer::opt_vec_to_motif(
            &optimized_result,
            &rec_db,
            &cfg.alpha,
            &cfg.max_count,
            &cfg.kmer,
        );
        optimized_motifs.push(optimized_motif);
    }

    motifer::pickle_motifs(
        &optimized_motifs,
        &cfg.out_fname,
    );
    println!("Vector of optimized motifs written to: {}", &cfg.out_fname);

    let motifs = motifer::filter_motifs(
        &mut optimized_motifs,
        &rec_db,
        &threshold,
        &cfg.max_count,
    );
    println!("{} motifs left after CMI-based filtering.", motifs.len());
}

