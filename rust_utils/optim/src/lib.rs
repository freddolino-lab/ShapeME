use rand::prelude::*;
use rand_distr::{Normal, Uniform};
//use rand_xoshiro::Xoshiro256PlusPlus;
use ordered_float::OrderedFloat;
//use std::sync::{Arc, Mutex};
use approx::assert_abs_diff_eq;
use approx::assert_abs_diff_ne;
use motifer;

//#[cfg(test)]
//mod tests {
//    use super::*;
//
//    /// Himmelblau test function (copied directly from argmin-testfunctions
//    /// source code then modified slightly)
//    ///
//    /// Defined as
//    ///
//    /// `f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2`
//    ///
//    /// where `x_i \in [-5, 5]`.
//    ///
//    /// The global minima are at
//    ///  * `f(x_1, x_2) = f(3, 2) = 0`.
//    ///  * `f(x_1, x_2) = f(-2.805118, 3.131312) = 0`.
//    ///  * `f(x_1, x_2) = f(-3.779310, -3.283186) = 0`.
//    ///  * `f(x_1, x_2) = f(3.584428, -1.848126) = 0`.
//    fn himmelblau(param: &Vec<f64>) -> f64 {
//        assert!(param.len() == 2);
//        let (x1, x2) = (param[0], param[1]);
//        (x1.powi(2) + x2 - 11.0).powi(2)
//            + (x1 + x2.powi(2) - 7.0).powi(2)
//    }
//
//    /// Multidimensional Rosenbrock test function (copied and slightly modified from
//    /// the argmin-testfunctions source)
//    ///
//    /// Defined as
//    ///
//    /// `f(x_1, x_2, ..., x_n) = \sum_{i=1}^{n-1} \left[ (a - x_i)^2 + b * (x_{i+1} - x_i^2)^2 \right]`
//    ///
//    /// where `x_i \in (-\infty, \infty)`. The parameters a and b usually are: `a = 1` and `b = 100`.
//    ///
//    /// The global minimum is at `f(x_1, x_2, ..., x_n) = f(1, 1, ..., 1) = 0`.
//    pub fn rosenbrock(param: &Vec<f64>) -> f64 {
//        param.iter()
//            .zip(param.iter().skip(1))
//            .map(|(&xi, &xi1)| (1.0 - xi).powi(2) + 100.0 * (xi1 - xi.powi(2)).powi(2))
//            .sum()
//    }
//
//    fn set_up_particle(
//            step: &f64,
//            low: &Vec<f64>,
//            up: &Vec<f64>,
//            objective: &dyn Fn(&Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64) -> f64
//    ) -> Particle {
//
//        let data_vec = vec![1.0, 1.0];
//        let T = 2.0;
//        let particle = Particle::new(
//            data_vec,
//            low.to_vec(),
//            up.to_vec(),
//            T,
//            *step,
//            objective,
//        );
//        particle
//    }
//
//    #[test]
//    fn test_eval() {
//        let step = 0.0;
//        let low = vec![-5.0, -5.0];
//        let up = vec![5.0, 5.0];
//        let particle = set_up_particle(&step, &low, &up, himmelblau);
//        let score = particle.evaluate();
//        assert_eq!(&score, &106.0);
//    }
//
//    #[test]
//    fn test_jitter_only() {
//        // Here we test that stepsize 0.0 and no velocity do not move particle
//        let start_data = vec![1.0, 1.0];
//        let step = 0.0;
//        let low = vec![-5.0, -5.0];
//        let up = vec![5.0, 5.0];
//        let mut particle = set_up_particle(&step, &low, &up, himmelblau);
//
//        particle.perturb();
//        // particle should have started at [1.0,1.0], and should not have moved
//        // with step of 0.0
//        particle.position.iter()
//            .zip(&start_data)
//            .for_each(|(a,b)| assert_abs_diff_eq!(a,b));
//
//        // Here we test that stepsize 1.0 does move particle
//        let step = 1.0;
//        let low = vec![-5.0, -5.0];
//        let up = vec![5.0, 5.0];
//        let mut particle = set_up_particle(&step, &low, &up, himmelblau);
//        particle.perturb();
//        // particle should end NOT at [1.0,1.0], so the sums should differ
//        assert_ne!(
//            particle.position.iter().sum::<f64>(),
//            start_data.iter().sum::<f64>(),
//        );
//    }
//
//    #[test]
//    fn test_velocity_only() {
//        // Here we test that stepsize 0.0 and velocity [1.0, 1.0] moves particle
//        // directionally
//        let start_data = vec![1.0, 1.0];
//        let step = 0.0;
//        let inertia = 1.0;
//        let local_weight = 0.5;
//        let global_weight = 0.5;
//        let global_best = vec![4.0, 4.0];
//        let low = vec![-5.0, -5.0];
//        let up = vec![5.0, 5.0];
//        let mut particle = set_up_particle(&step, &low, &up, himmelblau);
//        // particle velocity is set to [0.0, 0.0] initially
//        particle.velocity.iter()
//            .zip(&vec![0.0, 0.0])
//            .for_each(|(a,b)| assert_abs_diff_eq!(a,b));
//        //assert!(particle.velocity.abs_diff_eq(&array![0.0, 0.0], 1e-6));
//        particle.set_velocity(
//            &inertia,
//            &local_weight,
//            &global_weight,
//            &global_best,
//        );
//        // particle velocity should have changed
//        particle.velocity.iter()
//            .zip(&vec![0.0,0.0])
//            .for_each(|(a,b)| assert_abs_diff_ne!(a,b));
//        //assert!(!particle.velocity.abs_diff_eq(&array![0.0, 0.0], 1e-6));
//        particle.perturb();
//        particle.position.iter()
//            .zip(&start_data)
//            .for_each(|(a,b)| assert_abs_diff_ne!(a,b));
//        //assert!(!particle.position.abs_diff_eq(&start_data, 1e-6));
//    }
//
//    #[test]
//    fn test_annealing() {
//        let step = 1.0;
//        let low = vec![-5.0, -5.0];
//        let up = vec![5.0, 5.0];
//        let mut particle = set_up_particle(&step, &low, &up, rosenbrock);
//        let niter = 1000;
//        let t_adj = 0.1;
//        let opt_params = simulated_annealing(
//            &mut particle,
//            niter,
//            &t_adj,
//        );
//        println!("{:?}", opt_params);
//    }
//
//    //#[test]
//    //fn test_swarming() {
//
//    //    let low = array![-5.0, -5.0];
//    //    let up = array![5.0, 5.0];
//    //    let swarm = Swarm::new(
//    //        100, // n_particles: usize
//    //        &low, // upper_bounds: &'a Array<f64, Ix1>,
//    //        &up, //lower_bounds: &'a Array<f64, Ix1>,
//    //        rosenbrock, //objective: fn(&ArrayView<f64, Ix1>) -> f64)
//    //    );
//    //}
//}

pub struct Particle<'a> {
    position: Vec<f64>,
    prior_position: Vec<f64>,
    best_position: Vec<f64>,
    best_score: f64,
    score: f64,
    lower_bound: Vec<f64>,
    upper_bound: Vec<f64>,
    temperature: f64,
    velocity: Vec<f64>,
    prior_velocity: Vec<f64>,
    stepsize: f64,
    objective: &'a dyn Fn(&Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64) -> f64,
    // The idea for using an Arc<Mutex<_>> here is taken directly from an example
    // optimization from the argmin crate. They state that that using this for the
    // random number generator allows for thread safe interior mutability.
    // I honestly don't know if this will be necessary for us, but here we have it.
    //rng: Arc<Mutex<Xoshiro256PlusPlus>>,
    rng: ThreadRng,
}

impl Particle<'_> {
    pub fn new<'a>(data: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        temperature: f64,
        stepsize: f64,
        objective: &'a dyn Fn(&Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64) -> f64,
        rec_db: &motifer::RecordsDB,
        kmer: &usize,
        max_count: &i64,
        alpha: &f64,
    ) -> Particle<'a> {

        // initialize velocity for each parameter to zero
        let v = vec![0.0; data.len()];
        let pv = vec![0.0; data.len()];
        // copy of data to place something into prior_position
        let d = data.to_vec();
        let pr = data.to_vec();
        let mut particle = Particle {
            position: data,
            prior_position: d,
            best_position: pr,
            best_score: f64::INFINITY,
            score: f64::INFINITY,
            lower_bound: lower,
            upper_bound: upper,
            temperature: temperature,
            velocity: v,
            prior_velocity: pv,
            objective: objective,
            stepsize: stepsize,
            //rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy())),
            rng: thread_rng(),
        };
        particle.score = particle.evaluate(
            rec_db,
            kmer,
            max_count,
            alpha,
        );
        particle.best_score = particle.score;
        particle
    }

    fn iterate(
            &mut self, inertia: &f64,
            local_weight: &f64,
            global_weight: &f64,
            global_best_position: &Vec<f64>,
            t_adj: &f64,
            rec_db: &motifer::RecordsDB,
            kmer: &usize,
            max_count: &i64,
            alpha: &f64,
    ) {
        // set the new velocity
        self.set_velocity(
            inertia,
            local_weight,
            global_weight,
            global_best_position,
        );
        self.perturb();
        let score = self.evaluate(rec_db, kmer, max_count, alpha);
        // If we reject the move, revert to prior state and perturb again.
        if !self.accept(&score) {
            self.revert();
        } else {
            // Update prior [and possibly the best] score if we accepted the move
            self.update_scores(&score);
        }
        self.adjust_temp(t_adj);
    }
    
    /// Update score fields after accepting a move
    fn update_scores(&mut self, score: &f64) {
        self.score = *score;
        // if this was our best-ever score, update best_score and best_position
        if *score < self.best_score {
            self.best_score = *score;
            self.best_position = self.position.to_vec();
        }
    }
    
    /// Revert current position and velocity to prior values
    fn revert(&mut self) {
        self.position = self.prior_position.to_vec();
        self.velocity = self.prior_velocity.to_vec();
    }
    
    /// Gets the score for this Particle
    fn evaluate(
            &self,
            rec_db: &motifer::RecordsDB,
            kmer: &usize,
            max_count: &i64,
            alpha: &f64,
    ) -> f64 {
        // the parens are necessary here!
        (self.objective)(&self.position, rec_db, kmer, max_count, alpha)
    }

    /// Randomly chooses the index of position to update using jitter
    fn choose_param_index(&mut self) -> usize {
        let die = Uniform::from(0..self.position.len());
        die.sample(&mut self.rng)
    }

    /// Set the velocity of the Particle
    fn set_velocity(&mut self, inertia: &f64,
            local_weight: &f64, global_weight: &f64,
            global_best_position: &Vec<f64>) {
        
        // before we change the velocity, set prior velocity to current velocity
        // this will enable reversion to prior state if we later reject the move
        self.prior_velocity = self.velocity.to_vec();
        // set stochastic element of weights applied to local and global best pos
        // self.rng.gen samples from [0.0, 1.0)
        let r_arr: [f64; 2] = self.rng.gen();
        // set the new velocity
        self.velocity.iter_mut() // mutably iterate over current velocity
            .zip(&self.best_position) // in lockstep with this Particle's best position
            .zip(global_best_position) // and the global best position
            .zip(&self.position) // and the current position
            .for_each(|(((a, b), c), d)| { // a=vel, b=local_best, c=all_best, d=pos
                let term1 = inertia * *a;
                let term2 = local_weight * r_arr[0] * (b - d);
                let term3 = global_weight * r_arr[1] * (c - d);
                *a = term1 + term2 + term3
            })
    }

    /// Adjusts the position of the Particle
    /// Note that all [Particle]s are instantiated with a velocity of zero.
    /// Therefore, if your optimization algorith does not make use of velocity,
    /// the velocity is never adjusted away from zero, so adding it here does
    /// nothing. If your method *does* use velocity, then it will have adjusted
    /// the velocity such that adding it here has an effect on its position.
    /// Complementary to that, if you want only the velocity to affect particle
    /// position, but no random jitter, set stepsize to 0.0.
    /// Modifies self.position in place.
    fn perturb(&mut self) {
        // before we change the position, set prior position to current position
        // this will enable reversion to prior state if we later reject the move
        self.prior_position = self.position.to_vec();

        // which index will we be nudging?
        let idx = self.choose_param_index();
        // by how far will we nudge?
        let jitter = self.get_jitter();

        // nudge the randomly chosen index by jitter
        self.position[idx] += jitter;
        // add velocity element-wise to position
        self.position.iter_mut() // mutably iterate over each position
            .zip(&self.velocity) // in lockstep with velocity in each dimension
            .zip(&self.lower_bound) // and the lower bounds for each dim
            .zip(&self.upper_bound) // and the upper bounds for each dim
            .for_each(|(((a, b), c), d)| *a = (*a + b).clamp(*c, *d)) // update position
    }

    /// Sample once from a normal distribution with a mean of 0.0 and std dev
    /// of self.stepsize.
    fn get_jitter(&mut self) -> f64 {
        // sample from normal distribution one time
        /////////////////////////////////////////////////////
        // Keep this self.rng.lock().unrwrap() for now. the argmin people use it
        // They say it's necessary for thread-safe optims
        /////////////////////////////////////////////////////
        //let mut rng = self.rng.lock().unwrap();

        /////////////////////////////////////////////////////
        // I keep wanting to make thist distr a field of Particle, but shouldn't:
        // I want to be able to update stepsize as optimization proceeds
        /////////////////////////////////////////////////////
        let jitter_distr = Normal::new(0.0, self.stepsize).unwrap();
        jitter_distr.sample(&mut self.rng)
    }

    /// Determine whether to accept the new position, or to go back to prior
    /// position and try again. If score is greater that prior score,
    /// return true. If the score is less than prior score, determine whether
    /// to return true probabilistically using the following function:
    ///
    /// `exp(-(score - prior_score)/T)`
    ///
    /// where T is temperature.
    fn accept(&mut self, score: &f64) -> bool {
        // compare this score to prior score
        let diff = score - self.score;
        // if score < last score, accept
        if diff < 0.0 {
            true
        // if score > last score, decide probabilistically whether to accept
        } else {
            // this is the function used by scipy.optimize.basinhopping for P(accept)
            let accept_prob = (-diff/self.temperature).exp();
            if accept_prob > self.rng.gen() {
                true
            }
            else {
                false
            }
        }
    }

    /// Adjusts the temperature of the Particle
    ///
    /// Defined as
    ///
    /// ` T_{i+1} = T_i * (1.0 - t_{adj})`
    ///
    /// # Arguments
    ///
    /// * `t_adj` - fractional amount by which to decrease the temperature of
    ///    the particle. For instance, if current temp is 1.0 and t_adj is 0.2,
    ///    the new temperature will be 1.0 * (1.0 - 0.2) = 0.8
    fn adjust_temp(&mut self, t_adj: &f64) {
        self.temperature *= (1.0 - t_adj)
    }
}

struct Swarm<'a> {
    particles: Vec<Particle<'a>>,
    global_best: Vec<f64>,
}

//impl Swarm<'_> {
//    pub fn new<'a>(n_particles: usize,
//            upper_bounds: &'a Vec<f64>,
//            lower_bounds: &'a Vec<f64>,
//            objective: fn(&Vec<f64>) -> f64) -> Swarm<'a> {
//
//        let mut rng = thread_rng();
//        let mut particle_vec = Vec::new();
//        for _ in 0..n_particles {
//            let mut data_vec = vec![0.0; upper_bounds.len()];
//            data_vec.iter_mut()
//                .enumerate()
//                .for_each(|(i,a)| {
//                    *a = rng.gen_range(lower_bounds[i]..upper_bounds[i]);
//                });
//            let particle = Particle::new(
//                data_vec,
//                lower_bounds.to_vec(),
//                upper_bounds.to_vec(),
//                0.0, // set temp and stepsize to 0.0 to turn off annealing
//                0.0,
//                &objective,
//            );
//            particle_vec.push(particle);
//        }
//        particle_vec.sort_unstable_by_key(|a| OrderedFloat(a.score));
//        let best_pos = particle_vec[0].position.to_vec();
//        Swarm{
//            particles: particle_vec,
//            global_best: best_pos,
//        }
//    }
//
//    fn iterate(&mut self) {
//    }
//}

struct ReplicaExchanger<'a> {
    particles: Vec<Particle<'a>>,
    iter_switch: usize,
}

pub fn simulated_annealing(
        particle: &mut Particle,
        niter: usize,
        t_adj: &f64,
        rec_db: &motifer::RecordsDB,
        kmer: &usize,
        max_count: &i64,
        alpha: &f64,
) -> Vec<f64> {

    let inertia = 0.0;
    let local_weight = 0.0;
    let global_weight = 0.0;
    let global_best_position = vec![0.0; particle.position.len()];

    for _ in 0..niter {
        particle.iterate(
            &inertia,
            &local_weight,
            &global_weight,
            &global_best_position,
            t_adj,
            rec_db,
            kmer,
            max_count,
            alpha,
        );
    }
    particle.position.to_vec()
}

//fn particle_swarm(swarm: &mut Swarm,
//                  niter: usize,
//                  inertia: f64,
//                  local_weight = f64,
//                  global_weight = f64) -> Array<f64, Ix1> {
//
//    let inertia = 0.0;
//    let local_weight = 0.0;
//    let global_weight = 0.0;
//
//    for _ in 0..niter {
//        swarm.iterate(
//            &inertia,
//            &local_weight,
//            &global_weight,
//        );
//    }
//    swarm.global_best.to_owned()
//}
