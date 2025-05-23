#!/bin/bash

python sac_halfcheetah.py --lr=0.0009833543280434516 --discount-factor=0.98 --tau=0.008851313784063113 --num-steps=1000000
python sac_halfcheetah.py --lr=0.0009332189626030728 --discount-factor=0.98 --tau=0.009325749191536416 --num-steps=1000000
python sac_halfcheetah.py --lr=0.0006344571440300171 --discount-factor=0.98 --tau=0.002244255539895659 --num-steps=1000000