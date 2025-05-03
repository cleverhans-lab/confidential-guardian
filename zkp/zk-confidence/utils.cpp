#ifndef _ZKPOF_UTILS
#define _ZKPOF_UTILS

//#define DEBUG

#include <emp-zk/emp-zk.h>
#include <iostream>
#include <emp-tool/emp-tool.h>
#include "constant.cpp"
#include <algorithm>
#include <random>

using namespace emp;
using namespace std;




// appends a private Bit to an Integer for verified arithmetic operations
// 32 bits by default
Integer bit_to_int(Bit input, int int_sz=32) {
    Integer x = Integer(int_sz, 0, PUBLIC);
    x[0] = input;
    return x;
}

// if given a True Bit, returns an Int with binary rep 111...111
// if given a False Bit, returns an Int with binary rep 000...000
Integer bit_to_mask(Bit input, int int_sz=32) {
    Integer x = Integer(int_sz, 0, PUBLIC);
    for (int i=0; i<int_sz; ++i) {
        x[i] = input;
    }
    return x;
}

// returns the (biased representation) of the float's exponent bits
// as a 9 bit Integer
Integer get_float_exp(Float & input) {
    Integer exp(32, 0, PUBLIC);
    for (int i=0; i<8; ++i) {
        exp[i] = input[i+23];
    }
    return exp;
}

// return the first n bits of the mantissa as an Integer
Integer get_mantissa(Float & input, int first_n_bits) {
    Integer ret(32, 0, PUBLIC);
    for (int i=0; i<first_n_bits; ++i) {
        ret[i] = input[i+(23 - first_n_bits)];
    }
    return ret;
}

// return first n bits of the mantissa with 1 prepended
// (for the implicit 1 in the floating point rep)
Integer get_mant_prepend_one(Float & input, int first_n_bits) {
    Integer ret(32, 0, PUBLIC);
    for (int i=0; i<first_n_bits; ++i) {
        ret[i] = input[i+(23 - first_n_bits)];
    }
    Bit ONE(1, PUBLIC);
    ret[first_n_bits] = ONE;
    return ret;
}

// converts a float LESS THAN 1 to a fixed point number
// truncates, does not round
// undefined behavior if float is greater than 1
Integer float_prob_to_fp(Float & input, int fractional_bits) {
    Integer ret = get_mant_prepend_one(input, fractional_bits-1);
    Integer exp = get_float_exp(input); // 9 bit, biased representation
    Integer ONE_TWENTY_SIX(32, 126, PUBLIC);
    Integer shift = ONE_TWENTY_SIX - exp;
    ret = ret >> shift;
    return ret;
}

Integer int_to_fp(Integer & input, int fractional_bits) {
    return input * Integer(32, (1<<fractional_bits), PUBLIC);
}


Float bit_to_float(Bit input) {
    Float ret(0.0, PUBLIC);
    Float ONE(1.0, PUBLIC);
    for (int i=0; i<32; ++i) {
        ret[i] = ONE[i] & input;
        //cout << i << ": " << ONE[i].reveal() << "  " << input.reveal() << endl;
    }
    return ret;
}


void print_float_vec(vector<Float> & xs) {
    int n = xs.size();
    cout << "(";
    for (int i=0; i<n; ++i) {
        cout << " " << xs[i].reveal<double>() << " ";
    }
    cout << ")\n";
}

Integer float_argmax(vector<Float> & xs, Integer & argmax_output, Float & max_output) {
    Integer ret(32, 0, ALICE);
    size_t n = xs.size();
    Float curr_max = xs[0];
    for (int i=1; i<n; ++i) {
        Integer curr_ind(32, i, PUBLIC);
        Bit flag = (curr_max.less_than(xs[i]));
        ret = ret.select(flag, curr_ind);
        //#ifdef DEBUG
        //cout << "iteration " << i << ":" << endl;
        //cout << "curr_max: " << curr_max.reveal<double>() << endl;
        //cout << "xs[i]: " << xs[i].reveal<double>() << endl;
        //cout << "flag: " << flag.reveal() << endl;
        //cout << "ret (post select): " << ret.reveal<int>() << endl;
        //#endif
        Float t = bit_to_float(flag);
        Float not_t = bit_to_float(flag ^ Bit(1,PUBLIC));
        curr_max = curr_max * not_t + xs[i] * t;
    }
    argmax_output = ret;
    max_output = curr_max;
    return ret;
}

Integer find_bin(Float phat, int B, int index_sz) {
    float float_B = 1.0 * B;
    Integer curr(index_sz, 0, PUBLIC);
    for (int i=0; i<B; ++i) {
        Integer index(index_sz, i, PUBLIC);
        Float lb(i/float_B, PUBLIC);
        Float ub((i+1)/float_B, PUBLIC);
        Bit flag1 = lb.less_equal(phat);
        Bit flag2 = phat.less_than(ub);
        curr = curr.select(flag1 & flag2, index);
    }
    return curr;
}

// return v1 if select_bit == 0
// return v2 if select_bit == 1
Float float_select(Float & v1, Bit & select_bit, Float & v2) {
    Float ret(0.0, PUBLIC);
    Bit not_select = select_bit ^ Bit(1, PUBLIC);
    for (int i=0; i<32; ++i) {
        ret[i] = (v1[i] & not_select) | (v2[i] & select_bit);
    }
    return ret;
}

// helper function for unit testing, generates bit vectors that simulate class and success probability distributions
// predicted_outcomes and sensitive_attributes should be vector<Bit>s that have just been initialized
// a0_pos is the proportion of points with class a0 that will have predicted outcome 1
// a1_pos is same for points with class a1
// sa_split gives the proportion of points with class a0
// example usage:
/*    
vector<Bit> predicted_outcomes;
vector<Bit> sensitive_attributes;
const int NUM_POINTS = 20;
example_bit_vectors_DP(predicted_outcomes, sensitive_attributes, NUM_POINTS, 0.8, 0.35, 0.5);
*/
void example_bit_vectors_DP(vector<Bit> & predicted_outcomes, vector<Bit> & sensitive_attributes, int num_points, double a0_pos, double a1_pos, double sa_split, bool verbose=false) {
    int num_s0 = num_points * sa_split;
    int num_s1 = num_points - num_s0;
    int num_s0_pos = num_s0 * a0_pos;
    //int num_s0_neg = num_s0 - num_s0_pos;
    int num_s1_pos = num_s1 * a1_pos;
    //int num_s1_neg = num_s1 - num_s1_pos;

    int s0_count = 0;
    int s0_pos_count = 0;
    int s1_pos_count = 0;
    for (int i=0; i<num_points; i++) {
        if (s0_count < num_s0) {
            sensitive_attributes.push_back(Bit(0, ALICE));
            s0_count++;
            if (s0_pos_count < num_s0_pos) {
                predicted_outcomes.push_back(Bit(1, ALICE));
                s0_pos_count++;
            } else {
                predicted_outcomes.push_back(Bit(0, ALICE));
            } 
        } else {
            sensitive_attributes.push_back(Bit(1, ALICE));
            if (s1_pos_count < num_s1_pos) {
                predicted_outcomes.push_back(Bit(1, ALICE));
                s1_pos_count++;
            } else {
                predicted_outcomes.push_back(Bit(0, ALICE));
            }
        }
    }
    if (verbose) {
        cout << "predicted outcomes:   [";
        for (int i=0; i<num_points; ++i) {
            cout << " " << predicted_outcomes[i].reveal() << " ";
        }
        cout << "]\n";
        cout << "sensitive attributes: [";
        for (int i=0; i<num_points; ++i) {
            cout << " " << sensitive_attributes[i].reveal() << " ";
        }
        cout << "]\n";
    }
}


vector<Float> gen_dummy_vec(size_t sz, float v) {
    vector<Float> ret;
    for (int i=0; i<sz; ++i) {
        ret.push_back(Float(v, ALICE));
    }
    return ret;
}

vector< vector<Float> > gen_dummy_weights(size_t in_sz, size_t out_sz, float v) {
    vector< vector<Float> > ret;
    for (int i=0; i<out_sz; ++i) {
        vector<Float> t;
        for (int j=0; j<in_sz; ++j) {
            t.push_back(Float(v, ALICE));
        }
        ret.push_back(t);
    }
    return ret;
}


#endif