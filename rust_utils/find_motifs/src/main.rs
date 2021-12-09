use motifer;
use optim;
use std::env;
use std::time;
use std::collections::HashMap;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

// Run target/release/find_motifs ../test_data/shapes.npy ../test_data/y_vals.npy ../test_data/config.pkl ../test_data/test_output.pkl

fn main() {
    let args: Vec<String> = env::args().collect();
    let cfg = motifer::parse_config(&args);

    ThreadPoolBuilder::new().num_threads(cfg.cores).build_global().unwrap();

    let rec_db = motifer::RecordsDB::new_from_files(
        &cfg.shape_fname,
        &cfg.yvals_fname,
    );
    let mut seeds = rec_db.make_seed_vec(cfg.kmer, cfg.alpha);

    let threshold = motifer::set_initial_threshold(
        &seeds,
        &rec_db,
        &cfg.seed_sample_size,
        &cfg.records_per_seed,
        &cfg.windows_per_record,
        &cfg.kmer,
        &cfg.alpha,
        &cfg.thresh_sd_from_mean,
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
    let mut motifs = motifer::filter_seeds(
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

    // Optimization options
    ////////////////////////////////////////////////////////////
    // NOTE: needs placed into a config file or something //////
    ////////////////////////////////////////////////////////////
    let p_temp: f64 = 0.20;
    // set temp to difference between p_temp and 0.5 on logit scale
    let temp = ((0.5+p_temp)/(0.5-p_temp)).ln().abs();
    let step = 0.25;
    // initial particles are dropped onto the landscape at
    // init_position + Normal(0.0, init_jitter)
    let init_jitter = &step * 8.0;
    let inertia = 0.8;
    let local_weight = 0.2;
    let global_weight = 0.8;

    let n_particles = 20;
    let n_iter = 10000;
    let n_iter_exchange = 2;
    let t_adjust = 0.0005;

    //println!("Grabbing the top two motifs to optimize");
    motifs.par_iter_mut()
        .enumerate()
        //.filter(|(i,motif)| i < &2)
        .for_each(|(i,motif)|
    {

        // get the starting MI just to see how the optimization went
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


        //println!("Using replica exchange as the optimization method.");
        //let (optimized_result,optimized_score) = optim::replica_exchange(
        //    params,
        //    low,
        //    up,
        //    n_particles,
        //    n_iter_exchange,
        //    temp,
        //    step,
        //    n_iter,
        //    &t_adjust,
        //    &motifer::optim_objective,
        //    &rec_db,
        //    &cfg.kmer,
        //    &cfg.max_count,
        //    &cfg.alpha,
        //    &true,
        //);

        //println!("Using particle swarm as the optimization method.");
        //let (optimized_result,optimized_score) = optim::particle_swarm(
        //    params,
        //    low,
        //    up,
        //    n_particles,
        //    inertia,
        //    local_weight,
        //    global_weight,
        //    init_jitter,
        //    n_iter,
        //    &motifer::optim_objective,
        //    &rec_db,
        //    &cfg.kmer,
        //    &cfg.max_count,
        //    &cfg.alpha,
        //    &true,
        //);

        println!("Optimizing motif {} using simulated annealing as the optimization method.", i);
        let (optimized_result,optimized_score) = optim::simulated_annealing(
            params,
            low,
            up,
            temp,
            step,
            n_iter,
            &t_adjust,
            &motifer::optim_objective,
            &rec_db,
            &cfg.kmer,
            &cfg.max_count,
            &cfg.alpha,
            &true,
        );

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
// the problem is either here or in objective fn. I am only getting hits of either [0,0] or [0,1], NEVER [1,1]!!!!!!!!!!!!!!!!!
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
        let optimized_motif = motifer::opt_vec_to_motif(
            &optimized_result,
            &rec_db,
            &cfg.alpha,
            &cfg.max_count,
            &cfg.kmer,
        );
        let duration = now.elapsed().as_secs_f64() / 60.0;

        println!("Optimizing motif {} took {:?} minutes.\nIt started with an adjusted mutual information of {}, and ended with a value of {}.", i, duration, &start_mi, &optimized_motif.mi);

        *motif = optimized_motif;
    });

    println!("Filtering optimized motifs based on connditional mutual information.");
    let motifs = motifer::filter_motifs(
        &mut motifs,
        &rec_db,
        &threshold,
        &cfg.max_count,
    );
    println!("{} motifs left after CMI-based filtering.", motifs.len());

    motifer::pickle_motifs(
        &motifs,
        &cfg.out_fname,
    );
    println!("Vector of filtered, optimized motifs written to: {}", &cfg.out_fname);
}

