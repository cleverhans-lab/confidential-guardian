#ifndef _ZKPOF_LR
#define _ZKPOF_LR


#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "utils.cpp"

using namespace emp;
using namespace std;

Float _sigmoid(Float z) {
    Float ONE = Float(1, PUBLIC);
    Float ret = ONE / (ONE + z.exp());
    return ret;
}

Float _linear(vector<Float> & input, vector<Float> & weights) {
    Float ZERO = Float(0, PUBLIC);
    Float ret = inner_product(begin(weights), end(weights), begin(input), ZERO);
    return ret;
}

Float binary_LR(vector<Float> & input, vector<Float> & weights) {
    Float logit = _linear(input, weights);
    Float ret = _sigmoid(logit);
    return ret;
}

Float LR_logit(vector<Float> & input, vector<Float> & weights) {
    Float logit = _linear(input, weights);
    return logit;
}

Float ood_binary_LR(vector<Float> & input, vector<Float> & weights, Float & ood_threshold) {
    Float logit = _linear(input, weights);
    Bit is_above_ood_thresh = ood_threshold.less_equal(logit);
    cout << "Above Threshold? " << is_above_ood_thresh.reveal() << endl;
    Float ret = _sigmoid(logit);
    return ret;
}


vector<Float> _softmax(vector<Float> & logits, const int num_classes) {
    vector<Float> exp_logits;
    vector<Float> ret;
    Float sum = Float(0, PUBLIC);
    for (int c=0; c<num_classes; c++) {
        Float temp = logits[c].exp();
        sum = sum + temp;
        exp_logits.push_back(temp);
    }

    for (int c=0; c<num_classes; c++) {
        ret.push_back(exp_logits[c] / sum);
    }
    return ret;
}

vector<Float> softmax_LR(vector<Float> & input, vector<vector<Float>> & weights, Float & ood_threshold, const int num_classes) {
    Bit is_above_ood_thresh = Bit(0, PUBLIC);
    vector<Float> logits;
    for (int c=0; c<num_classes; c++) {
        cout << c << "\n";
        Float logit = _linear(input, weights[c]);
        logits.push_back(logit);
        is_above_ood_thresh = (ood_threshold.less_equal(logit)) | is_above_ood_thresh; // if any logits are above threshold, not ood
    }
    cout << "Above Threshold? " << is_above_ood_thresh.reveal() << endl;
    vector<Float> ret = _softmax(logits, num_classes);
    return ret;
}

// sz should be the size of the input vector
void _ReLU(vector<Float> & input, size_t sz) {
    Float ZERO = Float(0, PUBLIC);
    Bit TRU(true, PUBLIC);
    for (int i=0; i<sz; ++i){
        input[i] = input[i].If(input[i].less_than(ZERO), ZERO);
    }
}

void _batch_norm(vector<Float> & input, vector<Float> & bn_recip, vector<Float> & bn_subtracts, size_t sz){
    for (int i=0; i<sz; ++i) {
        input[i] = input[i] * bn_recip[i] - bn_subtracts[i];
    }
}

vector<Float> one_layer_softmax_NN(size_t input_sz, size_t hr_sz, size_t out_sz, vector<Float> & input, vector< vector< Float >> & hweights, vector< vector<Float> > & outweights){
    Float ZERO(0, PUBLIC);
    vector<Float> hr; // hidden representation
    for (int i=0; i<hr_sz; ++i) {
        hr.push_back(inner_product(begin(hweights[i]), end(hweights[i]), begin(input), ZERO));
    }

    #ifdef DEBUG
    /*
    cout << "Wx: ";
    print_float_vec(hr);
    */
    #endif

    _ReLU(hr, hr_sz);

    #ifdef DEBUG
    /*
    cout << "after ReLU: ";
    print_float_vec(hr);
    */
    #endif

    vector<Float> logits; 
    for (int i=0; i<out_sz; ++i) {
        logits.push_back(inner_product(begin(outweights[i]), end(outweights[i]), begin(hr), ZERO));
    }

    #ifdef DEBUG
    /*
    cout << "logits: ";
    print_float_vec(logits);
    */
    #endif

    vector<Float> out_prob = _softmax(logits, out_sz);

    #ifdef DEBUG
    /*
    cout << "prob: ";
    print_float_vec(out_prob);
    */
    #endif
    return out_prob;
}

// batch norm divisors should contain *reciprocals* of divisor terms
vector<Float> tabular_model(vector<Float> & input, size_t hr1_sz, vector< vector<Float> > & hweights1, vector<Float> & batch_norm_divisors1, vector<Float> & batch_norm_subtracts1, size_t hr2_sz, vector< vector<Float>> & hweights2, vector<Float> & batch_norm_divisors2, vector<Float> & batch_norm_subtracts2, size_t hr3_sz, vector< vector<Float>> hweights3) {
    Float ZERO(0, PUBLIC);
    vector<Float> hr1; // hidden representation
    for (int i=0; i<hr1_sz; ++i) {
        hr1.push_back(inner_product(begin(hweights1[i]), end(hweights1[i]), begin(input), ZERO));
    }

    _ReLU(hr1, hr1_sz);
    _batch_norm(hr1, batch_norm_divisors1, batch_norm_subtracts1, hr1_sz);

    vector<Float> hr2;
    for (int i=0; i<hr2_sz; ++i) {
        hr2.push_back(inner_product(begin(hweights2[i]), end(hweights2[i]), begin(hr1), ZERO));
    }

    _ReLU(hr2, hr2_sz);
    _batch_norm(hr2, batch_norm_divisors2, batch_norm_subtracts2, hr2_sz);

    vector<Float> hr3;
    for (int i=0; i<hr3_sz; ++i) {
        hr3.push_back(inner_product(begin(hweights3[i]), end(hweights3[i]), begin(hr2), ZERO));
    }
    _ReLU(hr3, hr3_sz);
    return hr3;
}



Bit threshold_uncertainty(Float & logit, Float & threshold) {
    Bit ret(PUBLIC, 0);
    ret = logit.less_than(threshold);
    return ret;
}

void verify_threshold_LR(vector<vector<Float>> & data, vector<Float> & weights, Float & threshold, int C, size_t num_points, size_t num_features) {
    #ifdef DEBUG
    if (weights.size() != num_features) {
        cout << "ERROR IN verify_threshold_LR: weight vector wrong size\n";
        cout << "weights.size(): " << weights.size() << "\t num_features: " << num_features << "\n";
        return;
    }
    if (data.size() != num_points) {
        cout << "ERROR IN verify_threshold_LR: data vector wrong size\n";
        cout << "data.size(): " << data.size() << "\t num_points: " << num_points << "\n";
        return;
    }
    for (size_t i=0; i<num_points; ++i) {
        if (data[i].size() != num_features) {
            cout << "ERROR IN verify_threshold_LR: data point " << i << " wrong size\n";
            cout << "data[" << i << "].size(): " << data[i].size() << "\t num_features: " << num_features << "\n";
            return;
        }
    }
    #endif

    Integer above_thresh_count(32, 0, PUBLIC); // initialize count to 0
    Float logit(0, ALICE);
    Integer temp(32,0,PUBLIC);
    for (size_t i=0; i<num_points; ++i) {
        logit = LR_logit(data[i], weights);
        temp = bit_to_int(threshold.less_than(logit)); 
        above_thresh_count = above_thresh_count + temp;
    }
    Integer desired_count(32, C, PUBLIC);
    Bit check = above_thresh_count == desired_count;
    cout << "check: " << check.reveal() << "\n";
}


#endif