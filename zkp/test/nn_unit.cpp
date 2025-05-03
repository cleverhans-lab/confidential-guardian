#include <emp-zk/emp-zk.h>
#include <emp-tool/emp-tool.h>
#include <iostream>
#include "zk-confidence/model_zk.cpp"


using namespace emp;
using namespace std;

int port, party;
const int threads = 12;


void test_circuit_zk(BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    //Integer a(32, 3, ALICE);
    //Integer b(32, 2, ALICE);
    //cout << (a - b).reveal<uint32_t>(PUBLIC) << endl;
    //Bit a(1, ALICE);
    //Bit b(0, ALICE);

    //Float x = bit_to_float(a);
    //Float y = bit_to_float(b);
    //cout << "should be 1: " << x.reveal<double>() << endl;
    //cout << "should be 0: " << y.reveal<double>() << endl;

    Float x(1.5, PUBLIC);
    for (int i=0; i<10; ++i) {
        cout << " " << x[i].reveal() << " ";
    }
    cout << "\n";

    Integer a(8, 3, PUBLIC);
    Integer s(4, 1, PUBLIC);
    Integer b = a >> s;
    cout << b.reveal<int>() << endl;


    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}


void one_layer_nn_unit(BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    size_t in_sz = 2;
    size_t hr_sz = 100;
    size_t out_sz = 3;
    vector< vector<Float> > W = gen_dummy_weights(in_sz, hr_sz, 1.0);
    vector< vector<Float> > U = gen_dummy_weights(hr_sz, out_sz, 0.01);
    
    vector<Float> x;
    x.push_back(Float(-1.0, ALICE));
    x.push_back(Float(2.0, ALICE));
    //for (int i=0; i<in_sz; ++i) {
    //    x.push_back(Float(1.0, ALICE));
    //}

    vector<Float> res = one_layer_softmax_NN(in_sz, hr_sz, out_sz,x, W, U);
    cout << "(";
    for (int i=0; i<out_sz; ++i) {
        cout << " " << res[i].reveal<double>() << " ";
    }
    cout << ")\n";
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}

void tabular_nn_unit(BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
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
    
    vector<Float> x;
    for (int i=0; i<in_sz; ++i) {
        x.push_back(Float(1.0, ALICE));
    }

    vector<Float> res = tabular_model(x, hr1_sz, W1, bnd1, bns1, hr2_sz, W2, bnd2, bns2, out_sz, W3);
    cout << "(";
    for (int i=0; i<out_sz; ++i) {
        cout << " " << res[i].reveal<double>() << " ";
    }
    cout << ")\n";
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}

void test_argmax(BoolIO<NetIO> *ios[threads], int party){
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    vector<Float> xs;
    xs.push_back(Float(0.5, ALICE));
    xs.push_back(Float(-2.3, ALICE));
    xs.push_back(Float(57.2, ALICE));
    xs.push_back(Float(21.0, ALICE));
    xs.push_back(Float(66.0, ALICE));
    xs.push_back(Float(1.0, ALICE));
    Integer ret_argmax;
    Float ret_max;
    float_argmax(xs, ret_argmax, ret_max);
    cout << ret_argmax.reveal<int>() << endl;
    cout << ret_max.reveal<double>() << endl;
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}

void test_bit_to_float(BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    //Integer a(32, 3, ALICE);
    //Integer b(32, 2, ALICE);
    //cout << (a - b).reveal<uint32_t>(PUBLIC) << endl;
    Bit a(1, ALICE);
    Bit b(0, ALICE);

    Float x = bit_to_float(a);
    Float y = bit_to_float(b);
    cout << "should be 1: " << x.reveal<double>() << endl;
    cout << "should be 0: " << y.reveal<double>() << endl;

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}

int main(int argc, char **argv) {
    parse_party_and_port(argv, &party, &port);
    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
    //test_circuit_zk(ios, party);
    //one_layer_nn_unit(ios, party);
    //test_argmax(ios, party);
    tabular_nn_unit(ios, party);

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}