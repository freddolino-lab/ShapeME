use std::path;
use std::env;
use std::io::BufReader;
use std::fs::File;
use serde::Deserialize;
use ndarray::prelude::*;
use info_theory;

#[derive(Deserialize)]
struct MotifsData {
    y: Vec<i64>,
    motifs: Vec<MotifData>,
    rec_num: u64,
}

#[derive(Deserialize)]
struct MotifData {
    hits: Vec<i64>,
    ami: f64,
    param_num: usize,
}

impl MotifsData {
    fn get_good_motif_indices(&self) -> String {

        // store indices of retained motifs
        let mut retained_inds: Vec<usize> = Vec::new();
        let mut result = String::new();
        let y_vals = Array::from_vec(self.y.to_vec());

        // Make sure first seed passes AIC
        let log_lik = self.rec_num as f64 * self.motifs[0].ami;
        let mut model_param_num: usize = self.motifs[0].param_num;

        // get aic, if it's less than zero, insert the index into the vec
        let aic = info_theory::calc_aic(model_param_num, log_lik);
        if aic < 0.0 {
            retained_inds.push(0);
            let res = 0.to_string();
            result.push_str(&res);
        } else {
            return String::new();
        }

        // loop through candidate motifs
        for (i,cand_motif) in self.motifs[1..self.motifs.len()].iter().enumerate() {
            let ind = i+1;

            // if this motif doesn't pass AIC on its own, with number of params, skip it
            let log_lik = self.rec_num as f64 * cand_motif.ami;
            if info_theory::calc_aic(cand_motif.param_num, log_lik) > 0.0 {
                continue
            }

            let cand_cats = Array::from_vec(cand_motif.hits.to_vec());
            let mut motif_pass = true;

            for motif_ind in retained_inds.iter() {

                // check the conditional mutual information for this motif with
                //   each of the retained motifs
                let good_cats = Array::from_vec(self.motifs[*motif_ind].hits.to_vec());

                let contingency = info_theory::construct_3d_contingency(
                    cand_cats.view(),
                    y_vals.view(),
                    good_cats.view(),
                );
                let cmi = info_theory::conditional_adjusted_mutual_information(
                    contingency.view()
                );

                // add candidate's parameter number to model
                model_param_num += cand_motif.param_num;
                let log_lik = self.rec_num as f64 * cmi;
                let this_aic = info_theory::calc_aic(model_param_num, log_lik);

                // if candidate motif doesn't improve model as added to each of the
                //   chosen motifs, skip it
                if this_aic > 0.0 {
                    // reverse the addition of the parameters for this candidate
                    model_param_num -= cand_motif.param_num;
                    break
                }
            }
            if motif_pass {
                retained_inds.push(ind);
                result.push(' ');
                let res = ind.to_string();
                result.push_str(&res);
            }
        }
        result
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let json_fname = &args[1];

    // read in motif info we need
    let file = File::open(json_fname).unwrap();
    // open a buffered reader to open the binary json file
    let buf_reader = BufReader::new(file);
    // hits/amis are sorted by descending ami in python script, no need to sort here
    let motif_data: MotifsData = serde_json::from_reader(buf_reader).unwrap();

    let result = motif_data.get_good_motif_indices();
    println!("{result}");
}
