// use ndarray::prelude::*;

#[cfg(test)]
mod tests {
    use super::*

    #[test]
    
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

/// An enumeration of parameter names
enum Params {
    EP,
    MGW,
    Roll,
    ProT,
    HelT,
}

/// struct for parameters
struct Param {
    name: Params, // name must be one of the enum Params
    vals: Vec<f32>, // vals is a vector of floating point 32-bit precision
}

/// For Motif, the idea here is that info has a key for each parameter.
///  The value associated with each parameter is a vector of tuples.
///  The first element of each tuple is the parameter's value, the second
///  is the weight.
struct Motif {
    info: Hashmap,
}

/// For Window, we have an info attribute that is simpler than that of Motif.
///  Window.info is a Hashmap, the keys of which are Params, and the values
///  of which are simple vectors of shape values.
struct Window {
    info: Hashmap,
}

/// Record contains a single piece of DNA's shape values and y-value
///  Also has a windows attribute, which is a vector of the windows
///  for the split up parameter values.
struct Record {
    params: Hashmap,
    windows: Vec<Hashmap>,
    y: u8,
}

impl Motif {
    /// Returns weighted distance between two Motif structs.
    ///
    /// # Examples
    ///
    /// ```
    /// ```
    fn constrained_manhattan_distance(
        &self,
        other: Window
    ) -> f32 {

        let mut dist = 0;

        for (param_name,param_info) in self.info {
            // normalize the weights here
            
            for pos in 0..len(param_info) {
                let x = param_info[pos][0];
                let w = 
                let y = other.get(param_name)[pos];
                let dist += abs(x - y) * w
            }
        }

        let w_exp = exp(w);
        let w = w_exp / sum(w_exp);
        let diff: Vec<f32> = 
            vec1.iter().zip(
                vec2.iter()
            ).map(|(&b, &v)| (b - v).abs()).collect();
        let dist = sum(diff * w);
        return dist
    }
}

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

/// Parses a CLI match from clap crate and converts to u32
pub fn fetch_int_arg(arg_val: &str) -> u32 {
    let int_val: u32 = arg_val.parse().unwrap();
    return int_val
}

/// Parses a CLI match from clap crate and converst to f32
pub fn fetch_float_arg(arg_val: &str) -> f32 {
    let float_val: f32 = arg_val.parse().unwrap();
    return float_val
}

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
