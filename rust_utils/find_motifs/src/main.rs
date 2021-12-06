use motifer;
use optim;
use std::env;
use std::time;
use std::collections::HashMap;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use approx::AbsDiff;

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
    for (i,motif) in motifs.iter().enumerate() {

        // for now, just do one motif
        if i == 1 {
            break;
        }
        println!("Optimizing motif {} now.", i);

        let start_mi = motif.mi;
        let now = time::Instant::now();
        let (params,low,up) = motifer::wrangle_params_for_optim(
            &motif,
            &shape_lb,
            &shape_ub,
            &weights_lb,
            &weights_ub,
            &thresh_lb,
            &thresh_ub,
        );

        let temp = 0.20;
        let step = 0.25;
        // initial particles are dropped onto the landscape at
        // init_position + Normal(0.0, init_jitter)
        let init_jitter = &step * 8.0;
        // set n_particles = 1 for simulated annealing of a single particle
        let n_particles = 50;
        let inertia = 0.8;
        let local_weight = 0.2;
        let global_weight = 0.8,

        let params_copy = params.to_vec();
        
        let n_iter = 1000;
        let t_adjust = 0.05;

        let (optimized_result,optimized_score) = optim::particle_swarm(
            params_copy,
            low,
            up,
            n_particles,
            inertia,
            local_weight,
            global_weight,
            init_jitter,
            n_iter,
            &motifer::optim_objective,
            &rec_db,
            &cfg.kmer,
            &cfg.max_count,
            &cfg.alpha,
        );
        
        //let (optimized_result,optimized_score) = optim::simulated_annealing(
        //    params_copy,
        //    low,
        //    up,
        //    temp,
        //    step,
        //    n_iter,
        //    &t_adjust,
        //    &motifer::optim_objective,
        //    &rec_db,
        //    &cfg.kmer,
        //    &cfg.max_count,
        //    &cfg.alpha,
        //);
        let optimized_motif = motifer::opt_vec_to_motif(
            &optimized_result,
            &rec_db,
            &cfg.alpha,
            &cfg.max_count,
            &cfg.kmer,
        );
        let duration = now.elapsed().as_secs_f64() / 60.0;
        println!("Optimizing motif {} took {:?} minutes.", i, duration);
        println!("It started with an adjusted mutual information of {}, and ended with a value of {}", &start_mi, &optimized_motif.mi);

        println!("==============================================");
        println!("Calculated MI for returned motif values was {}, and MI directly returned from optimizer was {}", &optimized_motif.mi, &optimized_score);
        println!("==============================================");

        assert!(AbsDiff::default()
            .epsilon(1e-6)
            .eq(&optimized_motif.mi,&optimized_score)
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

