use rand::prelude::*;
use rand_distr::{Normal, Uniform};
//use rand_xoshiro::Xoshiro256PlusPlus;
use ordered_float::OrderedFloat;
//use std::sync::{Arc, Mutex};
use approx::assert_abs_diff_eq;
use approx::assert_abs_diff_ne;
use rayon::prelude::*;
use ndarray::prelude::*;
use motifer;

#[cfg(test)]
mod tests {
    use super::*;

    /// Himmelblau test function (copied directly from argmin-testfunctions
    /// source code then modified slightly)
    ///
    /// Defined as
    ///
    /// `f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2`
    ///
    /// where `x_i \in [-5, 5]`.
    ///
    /// The global minima are at
    ///  * `f(x_1, x_2) = f(3, 2) = 0`.
    ///  * `f(x_1, x_2) = f(-2.805118, 3.131312) = 0`.
    ///  * `f(x_1, x_2) = f(-3.779310, -3.283186) = 0`.
    ///  * `f(x_1, x_2) = f(3.584428, -1.848126) = 0`.
    fn himmelblau(param: &Vec<f64>, this: &motifer::RecordsDB, is: &usize, contrived: &i64, i_know: &f64) -> f64 {
        assert!(param.len() == 2);
        let (x1, x2) = (param[0], param[1]);
        (x1.powi(2) + x2 - 11.0).powi(2)
            + (x1 + x2.powi(2) - 7.0).powi(2)
    }

    /// Multidimensional Rosenbrock test function (copied and slightly modified from
    /// the argmin-testfunctions source)
    ///
    /// Defined as
    ///
    /// `f(x_1, x_2, ..., x_n) = \sum_{i=1}^{n-1} \left[ (a - x_i)^2 + b * (x_{i+1} - x_i^2)^2 \right]`
    ///
    /// where `x_i \in (-\infty, \infty)`. The parameters a and b usually are: `a = 1` and `b = 100`.
    ///
    /// The global minimum is at `f(x_1, x_2, ..., x_n) = f(1, 1, ..., 1) = 0`.
    pub fn rosenbrock(param: &Vec<f64>, rec_db: &motifer::RecordsDB, nothing: &usize, nada: &i64, zip: &f64) -> f64 {
        param.iter()
            .zip(param.iter().skip(1))
            .map(|(&xi, &xi1)| (1.0 - xi).powi(2) + 100.0 * (xi1 - xi.powi(2)).powi(2))
            .sum()
    }

    fn set_up_recdb() -> motifer::RecordsDB {
        motifer::RecordsDB::new(
            vec![motifer::StrandedSequence::new(Array3::zeros((2,2,2)))],
            array![0],
        )
    }

    fn set_up_swarm<'a>(
            n_particles: usize,
            temp: f64,
            step: &'a f64,
            low: &'a Vec<f64>,
            up: &'a Vec<f64>,
            init_jitter: f64,
            objective: &'a dyn Fn(&Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64) -> f64
    ) -> Swarm<'a> {

        let data_vec = vec![0.0, 0.0];
        let rec_db = set_up_recdb();

        Swarm::new(
            n_particles,
            data_vec,
            low.to_vec(),
            up.to_vec(),
            temp,
            *step,
            init_jitter,
            objective,
            &rec_db,
            &2,
            &1,
            &0.01,
        )
    }

    fn set_up_particle(
            step: &f64,
            low: &Vec<f64>,
            up: &Vec<f64>,
            objective: &dyn Fn(&Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64) -> f64
    ) -> Particle {

        let rec_db = set_up_recdb();
        let data_vec = vec![0.0, 0.0];
        let T = 2.0;
        let particle = Particle::new(
            data_vec,
            low.to_vec(),
            up.to_vec(),
            objective,
            T,
            *step,
            &rec_db, // this and the next three args are just place-holders to make code compile so that I can use himmelblau and rosenbrock
            &15,
            &1,
            &0.1,
        );
        particle
    }

    #[test]
    fn test_eval() {
        let rec_db = set_up_recdb();
        let step = 0.0;
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let particle = set_up_particle(&step, &low, &up, &himmelblau);
        let score = particle.evaluate(
            &himmelblau,
            &rec_db,
            &5,
            &1,
            &0.01,
        );
        assert_eq!(&score, &170.0);
    }

    #[test]
    fn test_jitter_only() {
        // Here we test that stepsize 0.0 and no velocity do not move particle
        let start_data = vec![0.0, 0.0];
        let step = 0.0;
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let mut particle = set_up_particle(&step, &low, &up, &himmelblau);

        particle.perturb();
        // particle should have started at [1.0,1.0], and should not have moved
        // with step of 0.0
        particle.position.iter()
            .zip(&start_data)
            .for_each(|(a,b)| assert_abs_diff_eq!(a,b));

        // Here we test that stepsize 1.0 does move particle
        let step = 1.0;
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let mut particle = set_up_particle(&step, &low, &up, &himmelblau);
        particle.perturb();
        // particle should end NOT at [1.0,1.0], so the sums should differ
        assert_ne!(
            particle.position.iter().sum::<f64>(),
            start_data.iter().sum::<f64>(),
        );
    }

    #[test]
    fn test_velocity_only() {
        // Here we test that stepsize 0.0 and velocity [1.0, 1.0] moves particle
        // directionally
        let start_data = vec![0.0, 0.0];
        let step = 0.0;
        let inertia = 1.0;
        let local_weight = 0.5;
        let global_weight = 0.5;
        let global_best = vec![4.0, 4.0];
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let mut particle = set_up_particle(&step, &low, &up, &himmelblau);
        // particle velocity is set to [0.0, 0.0] initially
        particle.velocity.iter()
            .zip(&vec![0.0, 0.0])
            .for_each(|(a,b)| assert_abs_diff_eq!(a,b));
        //assert!(particle.velocity.abs_diff_eq(&array![0.0, 0.0], 1e-6));
        particle.set_velocity(
            &inertia,
            &local_weight,
            &global_weight,
            &global_best,
        );
        // particle velocity should have changed
        particle.velocity.iter()
            .zip(&vec![0.0,0.0])
            .for_each(|(a,b)| assert_abs_diff_ne!(a,b));
        //assert!(!particle.velocity.abs_diff_eq(&array![0.0, 0.0], 1e-6));
        particle.perturb();
        particle.position.iter()
            .zip(&start_data)
            .for_each(|(a,b)| assert_abs_diff_ne!(a,b));
        //assert!(!particle.position.abs_diff_eq(&start_data, 1e-6));
    }

    #[test]
    fn test_temp_switch() {
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    }

    #[test]
    fn test_annealing() {
        let rec_db = set_up_recdb();
        let step = 0.25;
        let start = vec![0.0, 0.0];
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let temp = 2.0;
        let niter = 10000;
        let t_adj = 0.01;

        let opt_params = simulated_annealing(
            start,
            low,
            up,
            temp,
            step,
            niter,
            &t_adj,
            &rosenbrock,
            &rec_db,
            &5,
            &1,
            &0.01,
        );

        println!("{:?}", opt_params);
    }

    #[test]
    fn test_swarming() {

        let rec_db = set_up_recdb();
        let step = 0.25;
        let start = vec![0.0, 0.0];
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let n_particles = 50;
        let inertia = 0.8;
        let local_weight = 0.2;
        let global_weight = 0.8;
        let initial_jitter = &step * 8.0;

        let niter = 1000;

        let opt_params = particle_swarm(
            start,
            low,
            up,
            n_particles,
            inertia,
            local_weight,
            global_weight,
            initial_jitter,
            niter,
            &rosenbrock,
            &rec_db,
            &5,
            &1,
            &0.01,
        );

        println!("{:?}", opt_params);
    }
}

pub enum Method {
    SimulatedAnnealing,
    ParticleSwarm,
    ReplicaExchange,
}

pub struct Particle {
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
    // The idea for using an Arc<Mutex<_>> here is taken directly from an example
    // optimization from the argmin crate. They state that that using this for the
    // random number generator allows for thread safe interior mutability.
    // I honestly don't know if this will be necessary for us, but here we have it.
    //rng: Arc<Mutex<Xoshiro256PlusPlus>>,
    rng: ThreadRng,
}

impl Particle {
    pub fn new(
        data: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        objective: &dyn Fn(&Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64) -> f64,
        temperature: f64,
        stepsize: f64,
        rec_db: &motifer::RecordsDB,
        kmer: &usize,
        max_count: &i64,
        alpha: &f64,
    ) -> Particle {

        // initialize velocity for each parameter to zero
        let mut v = vec![0.0; data.len()];
        for value in v.iter_mut() {
            *value = *value + thread_rng().gen_range(-stepsize..stepsize)
        }
        let pv = v.to_vec();
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
            stepsize: stepsize,
            //rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy())),
            rng: thread_rng(),
        };
        particle.score = particle.evaluate(
            objective,
            rec_db,
            kmer,
            max_count,
            alpha,
        );
        particle.best_score = particle.score;
        particle
    }

    fn step(
            &mut self,
            objective: &dyn Fn(
                &Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64
            ) -> f64,
            inertia: &f64,
            local_weight: &f64,
            global_weight: &f64,
            global_best_position: &Vec<f64>,
            t_adj: &f64,
            rec_db: &motifer::RecordsDB,
            kmer: &usize,
            max_count: &i64,
            alpha: &f64,
            method: &Method,
    ) {
        // set the new velocity and move particle
        self.set_velocity(
            inertia,
            local_weight,
            global_weight,
            global_best_position,
        );
        self.perturb();

        let score = self.evaluate(objective, rec_db, kmer, max_count, alpha);

        match method {
            // if we did particle swarm, just update and move on
            Method::ParticleSwarm => self.update_scores(&score),
            // If we reject the move, revert to prior state and perturb again.
            Method::SimulatedAnnealing => {
                if !self.accept(&score) {
                    self.revert();
                } else {
                    // Update prior [and possibly the best] score if we accepted the move
                    self.update_scores(&score);
                }
            }
            Method::ReplicaExchange => {
                if !self.accept(&score) {
                    self.revert();
                } else {
                    // Update prior [and possibly the best] score if we accepted the move
                    self.update_scores(&score);
                }
            }
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
            objective: &dyn Fn(
                &Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64
            ) -> f64,
            rec_db: &motifer::RecordsDB,
            kmer: &usize,
            max_count: &i64,
            alpha: &f64,
    ) -> f64 {
        // the parens are necessary here!
        (objective)(&self.position, rec_db, kmer, max_count, alpha)
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
        // if score < last score, diff is < 0.0, and we accept
        if diff <= 0.0 {
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

pub struct Swarm<'a> {
    particles: Vec<Particle>,
    global_best_position: Vec<f64>,
    global_best_score: f64,
    objective: &'a dyn Fn(&Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64) -> f64,
}

impl Swarm<'_> {
    /// Returns a Swarm of particles whose positions are sampled from
    /// a normal distribution defined by the original start position
    /// plus 1.5 * stepsize.
    pub fn new<'a>(
            n_particles: usize,
            data: Vec<f64>,
            lower: Vec<f64>,
            upper: Vec<f64>,
            temperature: f64,
            stepsize: f64,
            initial_jitter: f64,
            objective: &'a dyn Fn(
                &Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64
            ) -> f64,
            rec_db: &motifer::RecordsDB,
            kmer: &usize,
            max_count: &i64,
            alpha: &f64,
            method: Method,
    ) -> Swarm<'a> {
        // instantiate the random number generator
        let mut rng = thread_rng();
        // set variance of new particles around data to initial_jitter^2
        let distr = Normal::new(0.0, initial_jitter).unwrap();

        // instantiate a Vec
        let mut particle_vec = Vec::new();
        // instantiate particles around the actual data
        for i in 0..n_particles {
            let mut data_vec = data.to_vec();
            let mut temp = 0.0;
            // first particle should be right on data
            match method {
                Method::SimulatedAnnealing => {
                    temp = temperature;
                }
                Method::ReplicaExchange => {
                    temp = temperature + distr.sample(&mut rng);
                }
                Method::ParticleSwarm => {
                    temp = 0.0;
                }
            }
            if i == 0 {
                let particle = Particle::new(
                    data_vec,
                    lower.to_vec(),
                    upper.to_vec(),
                    objective,
                    temp,
                    stepsize,
                    rec_db,
                    kmer,
                    max_count,
                    alpha,
                );
                particle_vec.push(particle);
            } else {

                data_vec.iter_mut()
                    .enumerate()
                    .for_each(|(i,a)| {
                        // set new particle's data to data + sample, clamp between bounds
                        *a = *a + distr
                            .sample(&mut rng)
                            .clamp(lower[i],upper[i]);
                    });
                let particle = Particle::new(
                    data_vec,
                    lower.to_vec(),
                    upper.to_vec(),
                    objective,
                    temp,
                    stepsize,
                    rec_db,
                    kmer,
                    max_count,
                    alpha,
                );
                particle_vec.push(particle);
            }
        }
        // sort by -score, since our score is opposite of AMI
        particle_vec.sort_unstable_by_key(|a| OrderedFloat(-a.score));
        let best_pos = particle_vec[0].best_position.to_vec();
        let best_score = particle_vec[0].best_score;
        Swarm{
            particles: particle_vec,
            global_best_position: best_pos,
            global_best_score: best_score,
            objective: objective,
        }
    }

    fn step(
            &mut self,
            inertia: &f64,
            local_weight: &f64,
            global_weight: &f64,
            t_adj: &f64,
            rec_db: &motifer::RecordsDB,
            kmer: &usize,
            max_count: &i64,
            alpha: &f64,
            method: &Method,
    ) {
        for particle in self.particles.iter_mut() {
        //self.particles.par_iter_mut().for_each(|particle| {
            particle.step(
                &self.objective,
                &inertia,
                &local_weight,
                &global_weight,
                &self.global_best_position,
                t_adj,
                rec_db,
                kmer,
                max_count,
                alpha,
                method,
            );
        //});
        }
        self.particles.sort_unstable_by_key(|particle| OrderedFloat(-particle.score));
        self.global_best_position = self.particles[0].best_position.to_vec();
        self.global_best_score = self.particles[0].best_score;
    }

    pub fn exchange_odd(&mut self) {
        let mut iterator = Vec::new();
        let swap_num = self.len() / 2;
        for i in 0..swap_num {
            iterator.push((i*2,i*2+1));
        }
        self.particles.sort_unstable_by_key(|particle| OrderedFloat(particle.temperature));
        for swap_idxs in iterator {
            let temp_i = self.particles[swap_idxs.1].temperature;
            let temp_i_plus_one = self.particles[swap_idxs.0].temperature;
            self.particles[swap_idxs.0].temperature = temp_i;
            self.particles[swap_idxs.1].temperature = temp_i_plus_one;
        }
    }

    pub fn exchange_even(&mut self) {
        let mut iterator = Vec::new();
        let swap_num = self.len() / 2 - 1;
        for i in 0..swap_num {
            iterator.push((i*2,i*2+1));
        }
        self.particles.sort_unstable_by_key(|particle| OrderedFloat(particle.temperature));
        for swap_idxs in iterator {
            let temp_i = self.particles[swap_idxs.1].temperature;
            let temp_i_plus_one = self.particles[swap_idxs.0].temperature;
            self.particles[swap_idxs.0].temperature = temp_i;
            self.particles[swap_idxs.1].temperature = temp_i_plus_one;
        }
    }

    /// Returns the number of particles in the Swarm
    pub fn len(&self) -> usize {self.particles.len()}
}

pub fn replica_exchange(
        params: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        n_particles: usize,
        n_iter_exchange: usize,
        temp: f64,
        step: f64,
        niter: usize,
        t_adj: &f64,
        objective: &dyn Fn(&Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64) -> f64,
        rec_db: &motifer::RecordsDB,
        kmer: &usize,
        max_count: &i64,
        alpha: &f64,
) -> (Vec<f64>, f64) {

    // set inertia, local_weight, and global_weight to 0.0 to turn off velocities,
    // thus leaving only the jitter to affect particle position
    let inertia = 0.0;
    let local_weight = 0.0;
    let global_weight = 0.0;
    // set initial_jitter to 0.0 to place all particles exactly at data
    let initial_jitter = 0.0;

    let mut swarm = Swarm::new(
        n_particles,
        params,
        lower,
        upper,
        temp,
        step,
        initial_jitter,
        objective,
        rec_db,
        kmer,
        max_count,
        alpha,
        Method::ReplicaExchange,
    );

    let mut odds = true;
    for i in 0..niter {
        if i % n_iter_exchange == 0 {
            if odds {
                swarm.exchange_odd();
            } else {
                swarm.exchange_even();
            }
        }
        swarm.step(
            &inertia,
            &local_weight,
            &global_weight,
            &t_adj,
            &rec_db,
            &kmer,
            &max_count,
            &alpha,
            &Method::ReplicaExchange,
        );
    }
    (swarm.global_best_position.to_vec(), swarm.global_best_score)
}

pub fn particle_swarm(
        params: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        n_particles: usize,
        inertia: f64,
        local_weight: f64,
        global_weight: f64,
        initial_jitter: f64,
        niter: usize,
        objective: &dyn Fn(&Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64) -> f64,
        rec_db: &motifer::RecordsDB,
        kmer: &usize,
        max_count: &i64,
        alpha: &f64,
) -> (Vec<f64>, f64) {

    // turn off jitter, leaving only velocity to affect position
    let step = 0.0;
    let temp = 0.0;
    let t_adj = 0.0;

    let mut swarm = Swarm::new(
        n_particles,
        params,
        lower,
        upper,
        temp,
        step,
        initial_jitter,
        objective,
        rec_db,
        kmer,
        max_count,
        alpha,
        Method::ParticleSwarm,
    );

    for i in 0..niter {
        swarm.step(
            &inertia,
            &local_weight,
            &global_weight,
            &t_adj,
            &rec_db,
            &kmer,
            &max_count,
            &alpha,
            &Method::ParticleSwarm,
        );
    }
    (swarm.global_best_position.to_vec(), swarm.global_best_score)
}

pub fn simulated_annealing(
        params: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        temp: f64,
        step: f64,
        niter: usize,
        t_adj: &f64,
        objective: &dyn Fn(&Vec<f64>, &motifer::RecordsDB, &usize, &i64, &f64) -> f64,
        rec_db: &motifer::RecordsDB,
        kmer: &usize,
        max_count: &i64,
        alpha: &f64,
) -> (Vec<f64>, f64) {

    // set inertia, local_weight, and global_weight to 0.0 to turn off velocities,
    // thus leaving only the jitter to affect particle position
    let inertia = 0.0;
    let local_weight = 0.0;
    let global_weight = 0.0;

    let mut swarm = Swarm::new(
        1, // n_particles is 1 for simulated annealing
        params, // Vec<f64>
        lower,
        upper,
        temp,
        step,
        0.0, // initial_jitter is 0.0 to place particle exactly at data
        objective,
        rec_db,
        kmer,
        max_count,
        alpha,
        Method::SimulatedAnnealing,
    );

    for i in 0..niter {
        swarm.step(
            &inertia,
            &local_weight,
            &global_weight,
            &t_adj,
            &rec_db,
            &kmer,
            &max_count,
            &alpha,
            &Method::SimulatedAnnealing,
        );
    }
    (swarm.global_best_position.to_vec(), swarm.global_best_score)
}

