use std::cmp;
use std::collections::HashMap;
use itertools::Itertools;
// ndarray stuff
use ndarray::prelude::*;
use ndarray::Array;
// ndarray_stats exposes ArrayBase to useful methods for descriptive stats like min.
use ndarray_stats::QuantileExt;
// ln_gamma is used a lot in expected mutual information to compute log-factorials
use statrs::function::gamma::ln_gamma;
// parallelization utilities provided by rayon crate
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_unique() {
        let arr = array![0,1,1,1,1,0,2,0,0,2,2,3];
        let av = arr.view();
        let uniques = unique_cats(av);
        assert_eq!(uniques, vec![0, 1, 2, 3]);
        assert_eq!(uniques.len(), 4);
    }

    #[test]
    fn test_entropy() {
        let count_arr = array![1,1];
        assert!(AbsDiff::default().epsilon(1e-6).eq(
                &entropy(count_arr.view()), &0.6931472))
    }

    #[test]
    fn test_contingency() {
        let a = array![0,1,2,0,1,2,2,0];
        let av = a.view();
        let b = array![1,2,3,1,2,2,1,2];
        let bv = b.view();

        let answer = array![
            [2, 1, 0],
            [0, 2, 0],
            [1, 1, 1]
        ];

        let contingency = construct_contingency_matrix(av, bv);
        assert_eq!(contingency, answer)
    }

    #[test]
    fn test_ami() {
        let a = array![0,1,2,0,1,2,2,0,3];
        let av = a.view();
        let b = array![1,2,3,1,2,2,1,2,1];
        let bv = b.view();

        let contingency = construct_contingency_matrix(av, bv);

        let ami = adjusted_mutual_information(contingency.view());
        let ami_answer = -0.04605733444793936;
        assert!(AbsDiff::default().epsilon(1e-6).eq(&ami_answer, &ami));

        let a = array![0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5];
        let av = a.view();
        let b = array![0,0,0,1,1,1,5,5,5,3,3,3,8,8,8,9,9,9];
        let bv = b.view();

        let contingency = construct_contingency_matrix(av, bv);

        let ami = adjusted_mutual_information(contingency.view());
        let ami_answer = 1.0;
        assert!(AbsDiff::default().epsilon(1e-6).eq(&ami_answer, &ami));
    }

    #[test]
    fn test_mi() {
        let a = array![0,1,2,0,1,2,2,0,3];
        let av = a.view();
        let b = array![1,2,3,1,2,2,1,2,1];
        let bv = b.view();

        let contingency = construct_contingency_matrix(av, bv);
        let mi = mutual_information(contingency.view());
        let mi_answer = 0.3865874373531204;
        assert!(AbsDiff::default().epsilon(1e-6).eq(&mi_answer, &mi));
    }

    #[test]
    fn test_3d_contingency() {
        let a = array![0,1,2,0,1,2,2,0,3];
        let av = a.view();
        let b = array![1,2,3,1,2,2,1,2,1];
        let bv = b.view();
        let c = array![1,2,3,1,2,2,1,2,1];
        let cv = c.view();

        let contingency = construct_3d_contingency(av, bv, cv);
        let answer = array![
            // zero in array a
            [
                // one in array b | two in b | three in b
                [2, 0, 0], // one in c
                [0, 1, 0], // two in c
                [0, 0, 0], // three in c
            ],
            // one in array a
            [
                // one in array b | two in b | three in b
                [0, 0, 0], // one in c
                [0, 2, 0], // two in c
                [0, 0, 0], // three in c
            ],
            // two in array a
            [
                // one in array b | two in b | three in b
                [1, 0, 0], // one in c
                [0, 1, 0], // two in c
                [0, 0, 1], // three in c
            ],
            // three in array a
            [
                // one in array b | two in b | three in b
                [1, 0, 0], // one in c
                [0, 0, 0], // two in c
                [0, 0, 0], // three in c
            ],
        ];
        assert!(answer.abs_diff_eq(&contingency, 0));
    }

    #[test]
    fn test_cond_mi() {
        // NOTE: this test needs a little work to get the ground truth
        // and to make it a proper test, but I think it's working fine.
        let a = array![0,1,2,0,1,2,2,0,3,0,2,1,1];
        let av = a.view();
        let b = array![1,2,3,1,2,2,1,2,1,1,2,2,3];
        let bv = b.view();
        let c = array![1,2,1,3,2,3,2,3,1,2,1,2,2];
        let cv = c.view();

        let contingency = construct_contingency_matrix(av, bv);
        let ami = adjusted_mutual_information(contingency.view());
        println!("AMI: {:?}", ami);

        let contingency = construct_3d_contingency(av, bv, cv);
        let cmi = conditional_adjusted_mutual_information(contingency.view());
        println!("CMI: {:?}", cmi);
        
        let c = array![1,2,3,1,2,2,1,2,1,1,2,2,3];
        let cv = c.view();

        let contingency = construct_3d_contingency(av, bv, cv);
        let cmi = conditional_adjusted_mutual_information(contingency.view());
        println!("negative CMI: {:?}", cmi);
    }
}

/// Get the AIC
///
/// # Arguments
///
/// * `par_num` - number of parameters
/// * `rec_num` - number of records in the RecordsDB
/// * `mi` - adjusted mutual information
pub fn calc_aic(par_num: usize, rec_num: usize, mi: f64) -> f64 {
    2.0 * par_num as f64 - 2.0 * rec_num as f64 * mi
}

/// Calculates the mutual information between vec1 and vec2, conditioned
/// on the contents of vec3.
///
/// # Arguments
///
/// * `vec1` - view to a vector
/// * `vec2` - view to a vector
/// * `vec3` - view to a vector
pub fn conditional_adjusted_mutual_information(
    contingency: ndarray::ArrayView<usize, Ix3>
) -> f64 {
    
    let N = contingency.sum() as f64;
    // c is the final axis sums
    let c = contingency
        // sum over first axis, leaving a 2d matrix
        .sum_axis(ndarray::Axis(0))
        // convert elements to f64
        .mapv(|elem| (elem as f64))
        // sum over first axis, leaving a vector
        .sum_axis(ndarray::Axis(0));

    let mut cmi_vec = Vec::new();
    // iterate over the final axes of contingency array
    for (z,nz) in c.iter().enumerate() {
        let pz = nz / N;
        // slice the appropriate contingency matrix for calculating mi
        let this_mat = contingency.slice(s![..,..,z]);
        // calculate mutual information between 
        let this_mi = adjusted_mutual_information(this_mat);
        // place this cmi into the vector of cmis
        cmi_vec.push(pz * this_mi);
    }
    // get sum on non NaN values in the cmi_vector
    cmi_vec.iter()
        // remove the NaN values
        .filter(|elem| !elem.is_nan())
        // take sum or remaining values after NaNs are removed
        .sum::<f64>()
}

/// Creates a 3d contingency array from three vectors
///
/// # Arguments
///
/// * `vec1` - ArrayView to a vector containing assigned categories
/// * `vec2` - ArrayView to a vector containing assigned categories
/// * `vec3` - ArrayView to a vector containing categories to condition on
pub fn construct_3d_contingency(
    vec1: ndarray::ArrayView::<i64, Ix1>,
    vec2: ndarray::ArrayView::<i64, Ix1>,
    vec3: ndarray::ArrayView::<i64, Ix1>
) -> ndarray::Array3<usize> {
    
    // get the distinct values present in each vector
    let vec1_cats = unique_cats(vec1);
    let vec2_cats = unique_cats(vec2);
    let vec3_cats = unique_cats(vec3);

    // allocate the contingency array of appropriate size
    let mut contingency = ndarray::Array::zeros(
        (vec1_cats.len(), vec2_cats.len(), vec3_cats.len())
    );

    // zip the first two vectors into a vector of tuples
    let zipped: Vec<(i64,i64)> = vec1.iter()
        .zip(vec2)
        .map(|(a,b)| (*a,*b))
        .collect();
    // zip the third vector's values into the tuples in the first vector
    let all_zipped: Vec<(i64,i64,i64)> = zipped.iter()
        .zip(vec3)
        .map(|((a,b), c)| (*a, *b, *c))
        .collect();

    // iterate over the categories for each vector and assign the number
    // of elements with each vector's value in our contingency array.
    for i in 0..vec1_cats.len() {
        for j in 0..vec2_cats.len() {
            for k in 0..vec3_cats.len() {
                contingency[[i, j, k]] = all_zipped.iter()
                    .filter(|x| **x == (vec1_cats[i], vec2_cats[j], vec3_cats[k]))
                    .collect::<Vec<&(i64,i64,i64)>>()
                    .len();
            }
        }
    }
    contingency
}


/// Converts a 2D array of hit counts, sorts the hits on each strand so that
/// the smaller number of hits comes first, and calculates the dot product
/// of the sorted hits array and the vector [1, max_count+1]. The resulting
/// vector is the hits categories, and is returned from this function.
///
/// # Arguments
///
/// * `hit_arr` - a mutable 2D hits array
/// * `max_count` - a reference to the maximum number of hits that is counted
///      on a strand.
pub fn categorize_hits(hit_arr: &ndarray::Array<i64, Ix2>, max_count: &i64) -> ndarray::Array<i64, Ix1> {
    let a = arr1(&[1, max_count + 1]); 
    hit_arr.dot(&a)
}
 
/// Calculates the mutual information between the vectors that gave rise
/// to the contingency table passed as an argument to this function.
///
/// # Arguments
///
/// * `contingency` - view to a matrix containing counts in each joint category.
pub fn mutual_information(contingency: ndarray::ArrayView<usize, Ix2>) -> f64 {

    let N = contingency.sum() as f64;
    let (R,C) = contingency.dim();
    // a is the row sums
    let a = contingency.sum_axis(ndarray::Axis(1)).mapv(|elem| (elem as f64));
    // b is the column sums
    let b = contingency.sum_axis(ndarray::Axis(0)).mapv(|elem| (elem as f64));

    let mut mi_vec = Vec::new();

    for (i,ni) in a.iter().enumerate() {
        // probability of i
        let pi = ni / N;
        for (j,nj) in b.iter().enumerate() {
            // probability of i and j
            let pij = contingency[[i,j]] as f64 / N;
            // probability of j
            let pj = nj / N;

            if pij == 0.0 || pi == 0.0 || pj == 0.0 {
                mi_vec.push(0.0);
            } else {
                // pij * log(pij / (pi * pj))
                //   = pij * (log(pij) - log(pi * pj))
                //   = pij * (log(pij) - (log(pi) + log(pj)))
                //   = pij * (log(pij) - log(pi) - log(pj))
                mi_vec.push(pij * (pij.ln() - pi.ln() - pj.ln()));
            }
        }
    }
    mi_vec.iter()
        // remove NaN values that come from categories with zero counts
        .filter(|elem| !elem.is_nan())
        // sum what remains after NaN removal
        .sum::<f64>()
}

/// Creates a contingency matrix from two vectors
///
/// # Arguments
///
/// * `vec1` - ArrayView to a vector containing assigned categories
/// * `vec2` - ArrayView to a vector containing assigned categories
pub fn construct_contingency_matrix(
    vec1: ndarray::ArrayView::<i64, Ix1>,
    vec2: ndarray::ArrayView::<i64, Ix1>
) -> ndarray::Array2<usize> {
    
    let vec1_cats = unique_cats(vec1);
    let vec2_cats = unique_cats(vec2);

    let mut contingency = ndarray::Array::zeros((vec1_cats.len(), vec2_cats.len()));

    let zipped: Vec<(i64,i64)> = vec1.iter().zip(vec2).map(|(a,b)| (*a,*b)).collect();

    for i in 0..vec1_cats.len() {
        for j in 0..vec2_cats.len() {
            contingency[[i, j]] = zipped
                .iter()
                .filter(|x| **x == (vec1_cats[i], vec2_cats[j]))
                .collect::<Vec<&(i64,i64)>>()
                .len();
        }
    }
    contingency
}

/// Calculates the adjusted mutual information for two vectors.
/// As the number of categories in two vectors increases, the
/// expected mutual information between them increases, even
/// when the categories for both vectors are randomly assigned.
/// Adjusted mutual information accounts for expected mutual
/// information to ensure that the maximum mutual information,
/// regardless of the number of categories, is 1.0. The minimum
/// is centered on 0.0, with negative values being possible.
/// Adjusted mutual information was published in:
///
/// Vinh, Nguyen Xuan, Julien Epps, and James Bailey. 2009. “Information Theoretic Measures for Clusterings Comparison: Is a Correction for Chance Necessary?” In Proceedings of the 26th Annual International Conference on Machine Learning, 1073–80. ICML ’09. New York, NY, USA: Association for Computing Machinery.
///
/// # Arguments
///
/// * `contingency` - A 2D Array of joint counts for the categories in two vectors.
///     This contingency matrix comes from construct_contingency_matrix
pub fn adjusted_mutual_information(
    contingency: ndarray::ArrayView<usize, Ix2>
) -> f64 {

    let emi = expected_mutual_information(contingency);
    let mi = mutual_information(contingency);
    let counts_a = contingency.sum_axis(ndarray::Axis(1));
    let h_1 = entropy(counts_a.view());
    let counts_b = contingency.sum_axis(ndarray::Axis(0));
    let h_2 = entropy(counts_b.view());

    let mean_entropy = (h_1 + h_2) * 0.5;

    let numerator = mi - emi;
    let denominator = mean_entropy - emi;

    let ami = numerator / denominator;
    ami
}


/// Calculate the expected mutual information for two vectors. This function
/// is essentially translated directly from (https://github.com/scikit-learn/scikit-learn/blob/0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/metrics/cluster/_expected_mutual_info_fast.pyx).
///
/// # Arguments
///
/// * `contingency` - Contingency table for the joint counts for categories in
///     two vectors
pub fn expected_mutual_information(
    contingency: ndarray::ArrayView<usize, Ix2>
) -> f64 {

    
    let (R,C) = contingency.dim();
    let N = contingency.sum() as f64;
    // a is the row sums
    let a = contingency.sum_axis(ndarray::Axis(1)).mapv(|elem| (elem as f64));
    // b is the column sums
    let b = contingency.sum_axis(ndarray::Axis(0)).mapv(|elem| (elem as f64));

    let max_a = a.max().unwrap();
    let max_b = b.max().unwrap();
    
    // There are three major terms to the EMI equation, which are multiplied to
    // and then summed over varying nij values.
    // While nijs[0] will never be used, having it simplifies the indexing.
    let max_nij = max_a.max(*max_b) + 1.0;
    let mut nijs = ndarray::Array::<f64, Ix1>::range(0.0, max_nij, 1.0);
    nijs[0] = 1.0; // stops divide by zero errors. not used, so not an issue.

    // term1 is nij / N
    let term1 = &nijs / N;

    // term2 is log((N*nij) / (a * b))
    //    = log(N * nij) - log(a*b)
    //    = log(N) + log(nij) - log(a*b)
    // the terms calculated here are used in the summations below
    let log_a = a.mapv(|elem| elem.ln());
    let log_b = b.mapv(|elem| elem.ln());
    let log_Nnij = N.ln() + nijs.mapv(|elem| elem.ln());

    // term3 is large, and involves many factorials. Calculate these in log
    //  space to stop overflows.
    // numerator = ai! * bj! * (N - ai)! * (N - bj)!
    // denominator = N! * nij! * (ai - nij)! * (bj - nij)! * (N - ai - bj + nij)!
    let gln_a = a.mapv(|elem| ln_gamma(elem + 1.0));
    let gln_b = b.mapv(|elem| ln_gamma(elem + 1.0));
    let gln_Na = a.mapv(|elem| ln_gamma(N - elem + 1.0));
    let gln_Nb = b.mapv(|elem| ln_gamma(N - elem + 1.0));
    let gln_N = ln_gamma(N + 1.0);
    let gln_nij = nijs.mapv(|elem| ln_gamma(elem + 1.0));

    // start and end values for nij terms for each summation
    let mut start = ndarray::Array2::<usize>::zeros((a.len(), b.len()));
    let mut end = ndarray::Array2::<usize>::zeros((R,C));

    for (i,v) in a.iter().enumerate() {
        for (j,w) in b.iter().enumerate() {
            // starting index of nijs to use as start of inner loop later
            start[[i,j]] = cmp::max((v + w - N) as usize, 1);
            // ending index of nijs to use as end of inner loop later
            // add 1 because of way for loop syntax works
            end[[i,j]] = cmp::min(*v as usize, *w as usize) + 1;
        }
    }

    // emi is a summation over various values
    let mut emi: f64 = 0.0;
    for i in 0..R {
        for j in 0..C {
            for nij in start[[i,j]]..end[[i,j]] {
                let term2 = log_Nnij[nij] - log_a[i] - log_b[j];
                // terms in the numerator are positive, terms in denominator
                // are negative
                let gln = gln_a[i]
                    + gln_b[j]
                    + gln_Na[i]
                    + gln_Nb[j]
                    - gln_N
                    - gln_nij[nij]
                    - ln_gamma(a[i] - nij as f64 + 1.0)
                    - ln_gamma(b[j] - nij as f64 + 1.0)
                    - ln_gamma(N - a[i] - b[j] + nij as f64 + 1.0);
                let term3 = gln.exp();
                emi += term1[nij] * term2 * term3;
            }
        }
    }
    emi
}

/// Calculated the proportion of elements in a vector belonging to each
/// distinct value in the vector.
///
/// # Arguments
///
/// * `vec` - A vector containing i64 values.
pub fn get_probs(vec: ndarray::ArrayView::<i64, Ix1>) -> HashMap<i64, f64> {
    let N = vec.len();
    // get a hashmap, keys of which are distinct values in vec, values are
    //  the number of ocurrences of the value.
    let vec_counts = vec.iter().counts();
    let mut p_i = HashMap::new();
    // iterate over key,value pairs in vec_counts
    for (key,value) in vec_counts.iter() {
        p_i.insert(**key, *value as f64 / N as f64);
    }
    p_i
}

pub fn unique_cats(arr: ndarray::ArrayView<i64, Ix1>) -> Vec<i64> {
    arr.iter().unique().cloned().collect_vec()
}

/// Get entropy from a vector of counts per class
///
/// # Arguments
///
/// * `counts_vec` - a vector containing the number of times
///    each category ocurred in the original data. For instance,
///    if the original data were [0,1,1,0,2], counts_vec would be
///    [2,2,1], since two zero's, two one's, and one two were in
///    the original data.
pub fn entropy(counts_vec: ndarray::ArrayView<usize, Ix1>) -> f64{
    let mut entropy = 0.0;
    let N = counts_vec.sum() as f64;
    for (i,ni) in counts_vec.iter().enumerate() {
        let pi = *ni as f64 / N;
        if pi == 0.0 {
            entropy += 0.0
        } else {
            entropy += pi * (pi.ln());
        }
    }
    -entropy
}

