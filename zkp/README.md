# Confidential Guardian - ZKP implementation

### Installation
Our ZKP implementation uses emp-toolkit. Here are install instructions:
1. `wget https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py`
2. `python[3] install.py --deps --tool --ot --zk`
    * By default it will build for Release. -DCMAKE_BUILD_TYPE=[Release|Debug] option is also available.
    * No sudo? Change CMAKE_INSTALL_PREFIX.

3. make a directory for OT-data `mkdir data`


### Testing
To compile the code, use `cmake . && make`

Then to run a test use `./run <test binary>` -- this locally simulates separate processes for the prover and verifier which interact to execute the protocol.

e.g.: `./run bin/test_basic_zk` will test a basic ZK circuit with `emp-toolkit`.

`./run bin/test_zkp_calibration_benchmark` will execute a runtime benchmark for Confidential Guardian. Modify `main` in `test/zkp_calibration_benchmark` to choose different datasets, reference set sizes, etc.



