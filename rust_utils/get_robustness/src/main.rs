use info_theory;
use std::process;
use ndarray::prelude::*;
use ndarray_npy;
use ndarray_npy::ReadNpyError;
use std::path;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let yvals_file = &args[1];
    let hits_file = &args[2];

    let yvals: Result<Array1<i64>, ReadNpyError> = ndarray_npy::read_npy(yvals_file);
    let yvals = match yvals {
        Ok(file) => file,
        Err(error) => panic!("Problem opening the file: {:?}\nError: {:?}", yvals_file, error),
    };
    let hits: Result<Array1<i64>, ReadNpyError> = ndarray_npy::read_npy(hits_file);
    let hits = match hits {
        Ok(file) => file,
        Err(error) => panic!("Problem opening the file: {:?}\nError: {:?}", hits_file, error),
    };

    let contingency = info_theory::construct_contingency_matrix(
        yvals.view(),
        hits.view(),
    );
    let ami = info_theory::adjusted_mutual_information(contingency.view());
    let robustness = info_theory::info_robustness(yvals.view(), hits.view());
    let zscore = info_theory::info_zscore(yvals.view(), hits.view());

    println!("ami: {}, robustness: {:?}, zscore: {}", ami, robustness, zscore.0);
}

