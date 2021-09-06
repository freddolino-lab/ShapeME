use std::error::Error;
// use ndarray::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn set_up_motif(val: f32, size: usize) -> Sequence {
        let ep_param = Param::new(ParamType::EP, vec![val; size]).unwrap();
        let prot_param = Param::new(ParamType::ProT, vec![val; size]).unwrap();
        let helt_param = Param::new(ParamType::HelT, vec![val; size]).unwrap();
        let roll_param = Param::new(ParamType::Roll, vec![val; size]).unwrap();
        let mgw_param = Param::new(ParamType::MGW, vec![val; size]).unwrap();
        let mut seq_vec = Vec::new();
        seq_vec.push(ep_param);
        seq_vec.push(prot_param);
        seq_vec.push(helt_param);
        seq_vec.push(roll_param);
        seq_vec.push(mgw_param);

        // make the Motif struct
        let this_sequence = Sequence::new(seq_vec);
        
        this_sequence.unwrap()
    }
 

    #[test]
    fn test_const_dist() {
        let this_motif = set_up_motif(2.0, 10);
        let that_motif = set_up_motif(1.0, 10);
        assert_eq!(manhattan_distance(&this_motif, &that_motif), 1.0*10.0*5.0);
    }
}

//fn run_query_over_ref(
//    y_vals: Vec<f32>,
//    query_shapes: Array<f32>,
//    query_weights: Array<f32>,
//    threshold: f32,
//    reference: Array<f32>,
//    R: u32,
//    W: u32,
//    dist_func: &dyn Fn(Array<f32>, Array<f32>, Array<f32>) -> f32,
//    max_count: u32,
//) {
//}

/// An enumeration of possible parameter names
/// Supported parameters could be added here in the future
/// We could also generalize it to be just a String type to take
/// anyones name for a parameter
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum ParamType {
    EP,
    MGW,
    Roll,
    ProT,
    HelT,
}

/// container struct for parameters. This should be read only
pub struct Param {
    name: ParamType, // name must be one of the enumerated Params
    vals: Vec<f32>, // vals is a vector of floating point 32-bit precision
}

/// container struct for a sequence or combo of params. This should be read only
pub struct Sequence {
    params: Vec<Param>
}

/// For Motif, the idea here is that info has a key for each parameter.
///  The value associated with each parameter is a vector of tuples.
///  The first element of each tuple is the parameter's value, the second
///  is the weight.
pub struct Motif {
    seq: Sequence,
    weights: Vec<f32>,
}

///// For Window, we have an info attribute that is simpler than that of Motif.
/////  Window.info is a HashMap, the keys of which are Params, and the values
/////  of which are simple vectors of shape values.
//#[derive(Debug)]
//struct Window {
//    info: HashMap,
//}

///// Record contains a single piece of DNA's shape values and y-value
/////  Also has a windows attribute, which is a vector of the windows
/////  for the split up parameter values.
//#[derive(Debug)]
//struct Record {
//    windows: Vec<HashMap>,
//    y: u8,
//}

impl Sequence {
    pub fn new(params: Vec<Param>) -> Result<Sequence, Box<dyn Error>> {
        Ok(Sequence { params })
    }
}

impl Param {
    pub fn new(name: ParamType, vals: Vec<f32>) -> Result<Param, Box<dyn Error>> {
        Ok(Param {name, vals})
    }

    pub fn subtract(&self, other: &Param) -> Vec<f32> {
        if self.name != other.name {
            panic!("Can't subtract params of different types")
        } else {
            self.iter().zip(other).map(|(a, b)| a - b).collect()
        }
    }
    fn iter(&self) -> ParamIterator {
        ParamIterator{slice: self.vals.iter()}
    }
}

// Allow for iteration over the Param without directly accessing the
// values vector. We need lifetime annotations throughout to tell Rust
// not to drop the Param while its being iterated through
pub struct ParamIterator<'a>{slice: std::slice::Iter<'a, f32>}

// This makes iter work for the Param. 
impl<'a> Iterator for ParamIterator<'a> {
    type Item = &'a f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.slice.next()
    }
}

// This allows the syntatic sugar of 'for val in param' to work
impl<'a> IntoIterator for &'a Param {
    type Item = &'a f32;
    type IntoIter = ParamIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}


pub fn manhattan_distance(seq1: &Sequence, seq2: &Sequence) -> f32 {
    let mut distance: f32 = 0.0;
    for (p1, p2) in seq1.params.iter().zip(&seq2.params){
        distance += p1.subtract(&p2).iter().map(|x| x.abs()).sum::<f32>();
    }
    distance
}
        



 //impl Motif {
 //    /// Returns weighted distance between two Motif structs.
 //    ///
 //    /// # Examples
 //    ///
 //    /// ```
 //    /// ```
 //    fn constrained_manhattan_distance(
 //        &self,
 //        other: &Motif,
 //    ) -> f32 {
 //
 //        let mut dist: f32 = 0.0;
 //
 //        let mut dist_vec = Vec::new();
 //        //NOTE: needs type here
 //        let mut self_vals = Vec::new();
 //        let mut other_vals = Vec::new();
 //        let mut w_exp_vec = Vec::new();
 //        let mut w_exp_sum = 0;
 //
 //        for (param_name,param) in self.params {
 //            // normalize the weights here
 //            
 //            for pos in 0..param.vals.len() {
 //                let self_val = param.vals[pos];
 //                let w = param.weights[pos];
 //                let w_exp = math::exp(w);
 //                w_exp_vec.push(w_exp);
 //                w_exp_sum += w_exp;
 //
 //                let other_val = other.params.get(&param_name).vals[pos];
 //                let dist = math::abs(self_val - other_val);
 //                dist_vec.push(dist);
 //            }
 //        }
 //        
 //        for idx in 0..dist_vec.len() {
 //            let w_exp = w_exp_vec[idx] / w_exp_sum;
 //            dist += w_exp * dist_vec[idx];
 //        }
 //
 //        return dist
 //    }
 //}

// NOTE: we can start from this to implement subtraction of two motifs
// I'm basically thinking we have `sub` be a method of a struct, `Motif`.
// We wouldn't have the output be another `Motif` struct, as defined here,
// but at least this gives us a start.
//impl Sub for Motif {
//
//    fn sub(self, other: &Motif) -> f64 {
//        for (param_name, param_vals) in other.vals {
//            let self_vals = self.vals.get(&param_name)
//            let dist = // implement Sub for params
//        }
//    }
//}

//fn parse_all_args(mat: clap::ArgMatches) -> Args {
//    
//    let kmer = fetch_int_arg(
//        mat
//        .value_of("kmer")
//        .unwrap()
//    );
//
//    let thresh_perc = fetch_float_arg(
//        mat
//        .value_of("threshold_perc")
//        .unwrap()
//    );
//
//    let thresh_mat = fetch_float_arg(
//        mat
//        .value_of("threshold_perc")
//        .unwrap()
//    );
//
//    let cont = fetch_int_arg(
//        mat
//        .value_of("continuous")
//        .unwrap()
//    );
//
//    let inf_r = fetch_int_arg(
//        mat
//        .value_of("inforobust")
//        .unwrap()
//    );
//
//    let fr_jack = fetch_float_arg(
//        mat
//        .value_of("frackjack")
//        .unwrap()
//    );
//
//    let args = Args {
//        infile: mat.value_of("infile").unwrap().to_string(),
//        // NOTE: unwrap is not safe for args that are not required.
//        //  should do if let for args that are not required.
//        //param_files: mat.values_of("params").unwrap().collect(),
//        //param_names: mat.values_of("param_names").unwrap().collect(),
//        kmer: kmer,
//        threshold_perc: thresh_perc,
//        threshold_match: thresh_mat,
//        continuous: cont,
//        inforobust: inf_r,
//        fracjack: fr_jack,
//        out_pref: mat.value_of("out_pref").unwrap(),
//    };
//
//    return args
//}

///// Parses a CLI match from clap crate and converts to u32
//pub fn fetch_int_arg(arg_val: &str) -> u32 {
//    let int_val: u32 = arg_val.parse().unwrap();
//    return int_val
//}
//
///// Parses a CLI match from clap crate and converst to f32
//pub fn fetch_float_arg(arg_val: &str) -> f32 {
//    let float_val: f32 = arg_val.parse().unwrap();
//    return float_val
//}

//fn parse_cli_yaml(cfg_fname: &str) -> Args {
//
//    // get yaml file containing argument structure
//    let yaml = clap::load_yaml!(cfg_fname);
//    let matches = clap::App::from_yaml(yaml).get_matches();
//
//    let args = parse_all_args(matches);
//
//    return args
//}
