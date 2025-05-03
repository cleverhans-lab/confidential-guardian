#include <emp-zk/emp-zk.h>
#include <emp-tool/emp-tool.h>
#include <iostream>
#include "zk-confidence/model_zk.cpp"

//#define DEBUG

using namespace emp;
using namespace std;

int port, party;
const int threads = 12;


void test_circuit_zk(BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    Integer a(32, 3, ALICE);
    Integer b(32, 2, ALICE);
    cout << (a - b).reveal<uint32_t>(PUBLIC) << endl;

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}


uint64_t comm(BoolIO<NetIO> *ios[threads]) {
  uint64_t c = 0;
  for (int i = 0; i < threads; ++i)
    c += ios[i]->counter;
  return c;
}
uint64_t comm2(BoolIO<NetIO> *ios[threads]) {
  uint64_t c = 0;
  for (int i = 0; i < threads; ++i)
    c += ios[i]->io->counter;
  return c;
}

// benchmark for a toy network
// N is number of data points in validation set
// B is the number of bins in the calibration plot
void bench_zkp_calibration_single_layer(BoolIO<NetIO> *ios[threads], int party, int N, int B, int fp_threshold) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    uint64_t com1 = comm(ios);
    uint64_t com11 = comm2(ios);
    int fractional_bits = 5;  

    // set up single layer 100-dim neural net
    size_t in_sz = 2;
    size_t hr_sz = 100;
    size_t out_sz = 3;
    vector< vector<Float> > W = gen_dummy_weights(in_sz, hr_sz, 1.0);
    vector< vector<Float> > U = gen_dummy_weights(hr_sz, out_sz, 0.01);
    
    // STEP 1: Prove Predicted Probabilities
    vector<Integer> ys;
    vector<Integer> yhats;
    vector<Float> phats;
    for (int i=0; i<N; ++i) {
        if (i % 25 == 0) {
            cout << "STEP 1:   " << i << "/" << N << endl;
        }

        // dummy input data (runtime is same for all values)
        vector<Float> x;
        for (int i=0; i<in_sz; ++i) {
            x.push_back(Float(1.0, ALICE));
        }
        Integer y(32, 2, ALICE); // dummy true label
        vector<Float> res = one_layer_softmax_NN(in_sz, hr_sz, out_sz,x, W, U);


        ys.push_back(y);
        Integer yhat(32, 0, PUBLIC);
        Float phat(0.0, PUBLIC);
        float_argmax(res, yhat, phat);
        yhats.push_back(yhat);
        phats.push_back(phat);
    }
    #ifdef DEBUG
    cout << "(";
    for (int i=0; i<N; ++i) {
        cout << " " << yhats[i].reveal<int>() << " ";
    }
    cout << ")\n";
    cout << "(";
    for (int i=0; i<N; ++i) {
        cout << " " << phats[i].reveal<double>() << " ";
    }
    cout << ")\n";
    #endif
    

    // STEP 2: Prove Bin Membership
    
    int index_sz = 5, step_sz = 14, val_sz = 32;

    ZKRAM<BoolIO<NetIO>> *Bin =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    ZKRAM<BoolIO<NetIO>> *Conf =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    ZKRAM<BoolIO<NetIO>> *Acc =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    
    // initialize all bins to 0
    Integer ZERO(val_sz, 0, PUBLIC);
    for (int i=0; i<B; ++i) {
        Integer index(index_sz, i, PUBLIC);
        Bin->write(index,ZERO);
        Bin->refresh();
        Conf->write(index, ZERO);
        Conf->refresh();
        Acc->write(index, ZERO);
        Acc->refresh();
    }

    Integer ONE(32, 1, PUBLIC);
    for (int i=0; i<N; ++i) {
        Float phat = phats[i];
        Integer bin_ind = find_bin(phat, B, index_sz);
        Integer bin_val = Bin->read(bin_ind);
        Bin->refresh();
        Bin->write(bin_ind, bin_val + ONE);
        Bin->refresh();
        
        Integer is_correct = bit_to_int(yhats[i] == ys[i]);
        Integer acc_val = Acc->read(bin_ind);
        Acc->refresh();
        Acc->write(bin_ind, acc_val + is_correct);
        Acc->refresh();

        /*
        #ifdef DEBUG
        Integer fp_phat = float_prob_to_fp(phat, fractional_bits);
        cout << "i: " << i << "   phat: " << phat.reveal<double>() << "  fp_phat: " << fp_phat.reveal<int>() << "\n";
        #endif 
        */

        
        Integer fp_phat = float_prob_to_fp(phat, fractional_bits);
        Integer conf_val = Conf->read(bin_ind);
        Conf->refresh();
        Conf->write(bin_ind, conf_val + fp_phat);
        Conf->refresh();
    }

    
    // STEP 3: Compute Bin Statistics
    Bit pass_audit(1,PUBLIC);
    Integer fp_theta(32, fp_threshold, PUBLIC);
    for (int i=0; i<B; ++i) {
        Integer bin_ind(index_sz, i, PUBLIC);
        Integer bin_i = Bin->read(bin_ind);
        Bin->refresh();
        Integer bin_fp = int_to_fp(bin_i, fractional_bits);

        Integer acc_i = Acc->read(bin_ind);
        Acc->refresh();
        Integer acc_fp = int_to_fp(acc_i, fractional_bits);

        Integer conf_fp = Conf->read(bin_ind);
        Conf->refresh();


        Bit pass_bin = (fp_theta * bin_fp >= (acc_fp - conf_fp).abs());
        pass_audit = pass_audit & pass_bin;
    }
    cout << "pass audit?: " << pass_audit.reveal() << endl;
    
    Bin->check();
    Conf->check();
    Acc->check();
    delete Bin;
    delete Conf;
    delete Acc;
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
    uint64_t com2 = comm(ios) - com1;
    uint64_t com22 = comm2(ios) - com11;
    std::cout << "communication (B): " << com2 << std::endl;
    std::cout << "communication (B): " << com22 << std::endl;
}



// benchmark for Adult dataset
// N is number of data points in validation set
// B is the number of bins in the calibration plot
void bench_zkp_calibration_tabular_adult(BoolIO<NetIO> *ios[threads], int party, int N, int B, int fp_threshold) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    uint64_t com1 = comm(ios);
    uint64_t com11 = comm2(ios);
    int fractional_bits = 5;  

    // set up dummy values for tabular model
    size_t in_sz = 57;
    size_t hr1_sz = 64;
    size_t hr2_sz = 32;
    size_t out_sz = 2;
    vector< vector<Float> > W1 = gen_dummy_weights(in_sz, hr1_sz, 0.01);
    vector< vector<Float> > W2 = gen_dummy_weights(hr1_sz, hr2_sz, 0.01);
    vector< vector<Float> > W3 = gen_dummy_weights(hr2_sz, out_sz, 0.01);
    vector<Float> bnd1 = gen_dummy_vec(hr1_sz, 1.1);
    vector<Float> bns1 = gen_dummy_vec(hr1_sz, 0.001);
    vector<Float> bnd2 = gen_dummy_vec(hr2_sz, 1.1);
    vector<Float> bns2 = gen_dummy_vec(hr2_sz, 0.001);

    // STEP 1: Prove Predicted Probabilities
    vector<Integer> ys;
    vector<Integer> yhats;
    vector<Float> phats;
    for (int i=0; i<N; ++i) {
        if (i % 25 == 0) {
            cout << "STEP 1:   " << i << "/" << N << endl;
        }

        // dummy input data (runtime is same for all values)
        vector<Float> x;
        for (int i=0; i<in_sz; ++i) {
            x.push_back(Float(1.0, ALICE));
        }
        Integer y(32, 2, ALICE); // dummy true label
        vector<Float> res = tabular_model(x, hr1_sz, W1, bnd1, bns1, hr2_sz, W2, bnd2, bns2, out_sz, W3);


        ys.push_back(y);
        Integer yhat(32, 0, PUBLIC);
        Float phat(0.0, PUBLIC);
        float_argmax(res, yhat, phat);
        yhats.push_back(yhat);
        phats.push_back(phat);
    }
    #ifdef DEBUG
    cout << "(";
    for (int i=0; i<N; ++i) {
        cout << " " << yhats[i].reveal<int>() << " ";
    }
    cout << ")\n";
    cout << "(";
    for (int i=0; i<N; ++i) {
        cout << " " << phats[i].reveal<double>() << " ";
    }
    cout << ")\n";
    #endif
    

    // STEP 2: Prove Bin Membership
    
    int index_sz = 5, step_sz = 14, val_sz = 32;

    ZKRAM<BoolIO<NetIO>> *Bin =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    ZKRAM<BoolIO<NetIO>> *Conf =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    ZKRAM<BoolIO<NetIO>> *Acc =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    
    // initialize all bins to 0
    Integer ZERO(val_sz, 0, PUBLIC);
    for (int i=0; i<B; ++i) {
        Integer index(index_sz, i, PUBLIC);
        Bin->write(index,ZERO);
        Bin->refresh();
        Conf->write(index, ZERO);
        Conf->refresh();
        Acc->write(index, ZERO);
        Acc->refresh();
    }

    Integer ONE(32, 1, PUBLIC);
    for (int i=0; i<N; ++i) {
        Float phat = phats[i];
        Integer bin_ind = find_bin(phat, B, index_sz);
        Integer bin_val = Bin->read(bin_ind);
        Bin->refresh();
        Bin->write(bin_ind, bin_val + ONE);
        Bin->refresh();
        
        Integer is_correct = bit_to_int(yhats[i] == ys[i]);
        Integer acc_val = Acc->read(bin_ind);
        Acc->refresh();
        Acc->write(bin_ind, acc_val + is_correct);
        Acc->refresh();

        /*
        #ifdef DEBUG
        Integer fp_phat = float_prob_to_fp(phat, fractional_bits);
        cout << "i: " << i << "   phat: " << phat.reveal<double>() << "  fp_phat: " << fp_phat.reveal<int>() << "\n";
        #endif 
        */

        
        Integer fp_phat = float_prob_to_fp(phat, fractional_bits);
        Integer conf_val = Conf->read(bin_ind);
        Conf->refresh();
        Conf->write(bin_ind, conf_val + fp_phat);
        Conf->refresh();
    }

    
    // STEP 3: Compute Bin Statistics
    Bit pass_audit(1,PUBLIC);
    Integer fp_theta(32, fp_threshold, PUBLIC);
    for (int i=0; i<B; ++i) {
        Integer bin_ind(index_sz, i, PUBLIC);
        Integer bin_i = Bin->read(bin_ind);
        Bin->refresh();
        Integer bin_fp = int_to_fp(bin_i, fractional_bits);

        Integer acc_i = Acc->read(bin_ind);
        Acc->refresh();
        Integer acc_fp = int_to_fp(acc_i, fractional_bits);

        Integer conf_fp = Conf->read(bin_ind);
        Conf->refresh();


        Bit pass_bin = (fp_theta * bin_fp >= (acc_fp - conf_fp).abs());
        pass_audit = pass_audit & pass_bin;
    }
    cout << "pass audit?: " << pass_audit.reveal() << endl;
    
    Bin->check();
    Conf->check();
    Acc->check();
    delete Bin;
    delete Conf;
    delete Acc;
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
    uint64_t com2 = comm(ios) - com1;
    uint64_t com22 = comm2(ios) - com11;
    std::cout << "communication (B): " << com2 << std::endl;
    std::cout << "communication (B): " << com22 << std::endl;
}

// benchmark for Credit dataset
// N is number of data points in validation set
// B is the number of bins in the calibration plot
void bench_zkp_calibration_tabular_credit(BoolIO<NetIO> *ios[threads], int party, int N, int B, int fp_threshold) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    uint64_t com1 = comm(ios);
    uint64_t com11 = comm2(ios);
    int fractional_bits = 5;  

    // set up dummy values for tabular model
    size_t in_sz = 18;
    size_t hr1_sz = 64;
    size_t hr2_sz = 32;
    size_t out_sz = 2;
    vector< vector<Float> > W1 = gen_dummy_weights(in_sz, hr1_sz, 0.01);
    vector< vector<Float> > W2 = gen_dummy_weights(hr1_sz, hr2_sz, 0.01);
    vector< vector<Float> > W3 = gen_dummy_weights(hr2_sz, out_sz, 0.01);
    vector<Float> bnd1 = gen_dummy_vec(hr1_sz, 1.1);
    vector<Float> bns1 = gen_dummy_vec(hr1_sz, 0.001);
    vector<Float> bnd2 = gen_dummy_vec(hr2_sz, 1.1);
    vector<Float> bns2 = gen_dummy_vec(hr2_sz, 0.001);

    // STEP 1: Prove Predicted Probabilities
    vector<Integer> ys;
    vector<Integer> yhats;
    vector<Float> phats;
    for (int i=0; i<N; ++i) {
        if (i % 25 == 0) {
            cout << "STEP 1:   " << i << "/" << N << endl;
        }

        // dummy input data (runtime is same for all values)
        vector<Float> x;
        for (int i=0; i<in_sz; ++i) {
            x.push_back(Float(1.0, ALICE));
        }
        Integer y(32, 2, ALICE); // dummy true label
        vector<Float> res = tabular_model(x, hr1_sz, W1, bnd1, bns1, hr2_sz, W2, bnd2, bns2, out_sz, W3);


        ys.push_back(y);
        Integer yhat(32, 0, PUBLIC);
        Float phat(0.0, PUBLIC);
        float_argmax(res, yhat, phat);
        yhats.push_back(yhat);
        phats.push_back(phat);
    }
    #ifdef DEBUG
    cout << "(";
    for (int i=0; i<N; ++i) {
        cout << " " << yhats[i].reveal<int>() << " ";
    }
    cout << ")\n";
    cout << "(";
    for (int i=0; i<N; ++i) {
        cout << " " << phats[i].reveal<double>() << " ";
    }
    cout << ")\n";
    #endif
    

    // STEP 2: Prove Bin Membership
    
    int index_sz = 5, step_sz = 14, val_sz = 32;

    ZKRAM<BoolIO<NetIO>> *Bin =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    ZKRAM<BoolIO<NetIO>> *Conf =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    ZKRAM<BoolIO<NetIO>> *Acc =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    
    // initialize all bins to 0
    Integer ZERO(val_sz, 0, PUBLIC);
    for (int i=0; i<B; ++i) {
        Integer index(index_sz, i, PUBLIC);
        Bin->write(index,ZERO);
        Bin->refresh();
        Conf->write(index, ZERO);
        Conf->refresh();
        Acc->write(index, ZERO);
        Acc->refresh();
    }

    Integer ONE(32, 1, PUBLIC);
    for (int i=0; i<N; ++i) {
        Float phat = phats[i];
        Integer bin_ind = find_bin(phat, B, index_sz);
        Integer bin_val = Bin->read(bin_ind);
        Bin->refresh();
        Bin->write(bin_ind, bin_val + ONE);
        Bin->refresh();
        
        Integer is_correct = bit_to_int(yhats[i] == ys[i]);
        Integer acc_val = Acc->read(bin_ind);
        Acc->refresh();
        Acc->write(bin_ind, acc_val + is_correct);
        Acc->refresh();

        /*
        #ifdef DEBUG
        Integer fp_phat = float_prob_to_fp(phat, fractional_bits);
        cout << "i: " << i << "   phat: " << phat.reveal<double>() << "  fp_phat: " << fp_phat.reveal<int>() << "\n";
        #endif 
        */

        
        Integer fp_phat = float_prob_to_fp(phat, fractional_bits);
        Integer conf_val = Conf->read(bin_ind);
        Conf->refresh();
        Conf->write(bin_ind, conf_val + fp_phat);
        Conf->refresh();
    }

    
    // STEP 3: Compute Bin Statistics
    Bit pass_audit(1,PUBLIC);
    Integer fp_theta(32, fp_threshold, PUBLIC);
    for (int i=0; i<B; ++i) {
        Integer bin_ind(index_sz, i, PUBLIC);
        Integer bin_i = Bin->read(bin_ind);
        Bin->refresh();
        Integer bin_fp = int_to_fp(bin_i, fractional_bits);

        Integer acc_i = Acc->read(bin_ind);
        Acc->refresh();
        Integer acc_fp = int_to_fp(acc_i, fractional_bits);

        Integer conf_fp = Conf->read(bin_ind);
        Conf->refresh();


        Bit pass_bin = (fp_theta * bin_fp >= (acc_fp - conf_fp).abs());
        pass_audit = pass_audit & pass_bin;
    }
    cout << "pass audit?: " << pass_audit.reveal() << endl;
    
    Bin->check();
    Conf->check();
    Acc->check();
    delete Bin;
    delete Conf;
    delete Acc;
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
    uint64_t com2 = comm(ios) - com1;
    uint64_t com22 = comm2(ios) - com11;
    std::cout << "communication (B): " << com2 << std::endl;
    std::cout << "communication (B): " << com22 << std::endl;
}

// take calibration benchmarks without inference, used to estimate runtime for larger neural networks by combining w/ runtime from Mystique
// N is number of data points in validation set
// B is the number of bins in the calibration plot
void bench_zkp_calibration_no_inf(BoolIO<NetIO> *ios[threads], int party, int N, int B, int fp_threshold) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    int fractional_bits = 5;  

    // set up
    size_t in_sz = 10; // dummy value, accounted for in mystique runtime
    //size_t hr1_sz = 64;
    //size_t hr2_sz = 32;
    size_t out_sz = 12;
    //vector< vector<Float> > W1 = gen_dummy_weights(in_sz, hr1_sz, 0.01);
    //vector< vector<Float> > W2 = gen_dummy_weights(hr1_sz, hr2_sz, 0.01);
    //vector< vector<Float> > W3 = gen_dummy_weights(hr2_sz, out_sz, 0.01);
    //vector<Float> bnd1 = gen_dummy_vec(hr1_sz, 1.1);
    //vector<Float> bns1 = gen_dummy_vec(hr1_sz, 0.001);
    //vector<Float> bnd2 = gen_dummy_vec(hr2_sz, 1.1);
    //vector<Float> bns2 = gen_dummy_vec(hr2_sz, 0.001);

    // STEP 1: Prove Predicted Probabilities
    vector<Integer> ys;
    vector<Integer> yhats;
    vector<Float> phats;
    for (int i=0; i<N; ++i) {
        if (i % 25 == 0) {
            cout << "STEP 1:   " << i << "/" << N << endl;
        }

        // dummy input data (runtime is same for all values)
        vector<Float> x;
        for (int i=0; i<in_sz; ++i) {
            x.push_back(Float(1.0, ALICE));
        }
        Integer y(32, 2, ALICE); // dummy true label
        vector<Float> res; // tabular_model(x, hr1_sz, W1, bnd1, bns1, hr2_sz, W2, bnd2, bns2, out_sz, W3);
        for (int i=0; i<out_sz; ++i) {
            res.push_back(Float(1.0, ALICE)); // dummy outputs w/ no inference for mystique estimated runtime
        }

        ys.push_back(y);
        Integer yhat(32, 0, PUBLIC);
        Float phat(0.0, PUBLIC);
        float_argmax(res, yhat, phat);
        yhats.push_back(yhat);
        phats.push_back(phat);
    }
    #ifdef DEBUG
    cout << "(";
    for (int i=0; i<N; ++i) {
        cout << " " << yhats[i].reveal<int>() << " ";
    }
    cout << ")\n";
    cout << "(";
    for (int i=0; i<N; ++i) {
        cout << " " << phats[i].reveal<double>() << " ";
    }
    cout << ")\n";
    #endif
    

    // STEP 2: Prove Bin Membership
    
    int index_sz = 5, step_sz = 14, val_sz = 32;

    ZKRAM<BoolIO<NetIO>> *Bin =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    ZKRAM<BoolIO<NetIO>> *Conf =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    ZKRAM<BoolIO<NetIO>> *Acc =
      new ZKRAM<BoolIO<NetIO>>(party, index_sz, step_sz, val_sz);
    
    // initialize all bins to 0
    Integer ZERO(val_sz, 0, PUBLIC);
    for (int i=0; i<B; ++i) {
        Integer index(index_sz, i, PUBLIC);
        Bin->write(index,ZERO);
        Bin->refresh();
        Conf->write(index, ZERO);
        Conf->refresh();
        Acc->write(index, ZERO);
        Acc->refresh();
    }

    Integer ONE(32, 1, PUBLIC);
    for (int i=0; i<N; ++i) {
        Float phat = phats[i];
        Integer bin_ind = find_bin(phat, B, index_sz);
        Integer bin_val = Bin->read(bin_ind);
        Bin->refresh();
        Bin->write(bin_ind, bin_val + ONE);
        Bin->refresh();
        
        Integer is_correct = bit_to_int(yhats[i] == ys[i]);
        Integer acc_val = Acc->read(bin_ind);
        Acc->refresh();
        Acc->write(bin_ind, acc_val + is_correct);
        Acc->refresh();

        /*
        #ifdef DEBUG
        Integer fp_phat = float_prob_to_fp(phat, fractional_bits);
        cout << "i: " << i << "   phat: " << phat.reveal<double>() << "  fp_phat: " << fp_phat.reveal<int>() << "\n";
        #endif 
        */

        
        Integer fp_phat = float_prob_to_fp(phat, fractional_bits);
        Integer conf_val = Conf->read(bin_ind);
        Conf->refresh();
        Conf->write(bin_ind, conf_val + fp_phat);
        Conf->refresh();
    }

    
    // STEP 3: Compute Bin Statistics
    Bit pass_audit(1,PUBLIC);
    Integer fp_theta(32, fp_threshold, PUBLIC);
    for (int i=0; i<B; ++i) {
        Integer bin_ind(index_sz, i, PUBLIC);
        Integer bin_i = Bin->read(bin_ind);
        Bin->refresh();
        Integer bin_fp = int_to_fp(bin_i, fractional_bits);

        Integer acc_i = Acc->read(bin_ind);
        Acc->refresh();
        Integer acc_fp = int_to_fp(acc_i, fractional_bits);

        Integer conf_fp = Conf->read(bin_ind);
        Conf->refresh();


        Bit pass_bin = (fp_theta * bin_fp >= (acc_fp - conf_fp).abs());
        pass_audit = pass_audit & pass_bin;
    }
    cout << "pass audit?: " << pass_audit.reveal() << endl;
    
    Bin->check();
    Conf->check();
    Acc->check();
    delete Bin;
    delete Conf;
    delete Acc;
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}

int main(int argc, char **argv) {
    auto start = clock_start();
    parse_party_and_port(argv, &party, &port);
    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
    //test_circuit_zk(ios, party);
    int fractional_bits = 5;
    int32_t fp_threshold = 0.15 * (1 << fractional_bits); // threshold encoded as a fixed point number
    int N = 100; // number of data points

    // runtime measured across entire execution of main, so uncomment one of the following lines depending on what benchmark you would like:
    //bench_zkp_calibration_single_layer(ios, party, N, 10, fp_threshold); // toy dataset
    bench_zkp_calibration_tabular_adult(ios, party, N, 10, fp_threshold); // Adult dataset
    //bench_zkp_calibration_tabular_credit(ios, party, N, 10, fp_threshold); // Credit dataset
    //bench_zkp_calibration_no_inf(ios, party, N, 10, fp_threshold); 
    // ^ other components w/o inference, used to estimate runtime for larger NNs by combining w/ Mystique runtime

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    double runtime = emp::time_from(start);
    cout << "N: " << N << "   runtime: " << runtime << endl;
    return 0;
}