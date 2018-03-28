# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 13:15:01 2018

@author: zrssch
"""
import numpy as np
import os
import pandas as pd

resa = pd.read_csv('submit_a.csv')
resb = pd.read_csv('submit_b.csv')
resc = pd.read_csv('submit_c.csv')
resd = pd.read_csv('submit_d.csv')

res1 = pd.read_csv('submit0303.csv')
res2 = pd.read_csv('submit0305.csv')
res3 = pd.read_csv('submit0306.csv')
res4 = pd.read_csv('submit0307.csv')
res5 = pd.read_csv('submit0307-2.csv')
res6 = pd.read_csv('submit0308.csv')
res7 = pd.read_csv('submit0309.csv')
res8 = pd.read_csv('submit0310.csv')
res9 = pd.read_csv('submit0311-2.csv')
res10 = pd.read_csv('submit0312.csv')
res11 = pd.read_csv('submit0312-2.csv')
CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

submit = pd.DataFrame(res1['id'])
for i in CLASSES:
    # submit[i] = ((res1[i]*res2[i]*res3[i]*res4[i]*res5[i]*res6[i]*res7[i]*res8[i]*res9[i]*res10[i])**(1.0/10)).values
    submit[i] = ((res1[i]*res2[i]*res3[i]*res5[i]*res8[i]*resa[i]*resb[i]*resc[i]*resd[i]*res9[i]*res10[i]*res11[i])**(1.0/12)).values
    # submit[i] = ((res1[i]*res3[i]*res8[i]*resa[i]*resb[i]*resc[i]*res9[i])/7).values
submit.to_csv('esem0312-2.csv', index=False)

'''
esem0310-2   0.9829    res1[i]*res2[i]*res3[i]*res5[i]*res8[i]*resa[i]*resb[i]*resc[i]*resd[i]    
esem0310-3   0.9828    res1[i]*res2[i]*res3[i]*res5[i]*resa[i]*resb[i]*resc[i]*resd[i]  
esem0311     0.9829    res1[i]*res2[i]*res3[i]*res5[i]*res8[i]*resa[i]*resb[i]*resc[i]*resd[i]*res9[i]      
esem0311-2   0.9829    (res1[i]*res2[i]*res3[i]*res5[i]*res8[i]*resa[i]*resb[i]*resc[i]*resd[i]*res9[i])/10 
esem0311-3   0.9828    (res1[i]*res3[i]*res8[i]*resa[i]*resb[i]*resc[i]*res9[i])/7   
esem0311-4   0.9828    (res1[i]*res3[i]*res8[i]*resa[i]*resb[i]*resc[i]*res9[i])**(1.0/7)
esem0312     0.9829    (res1[i]*res2[i]*res3[i]*res5[i]*res8[i]*resa[i]*resb[i]*resc[i]*resd[i]*res9[i]*res10[i])**(1.0/11)
esem0312-2   0.9829    (res1[i]*res2[i]*res3[i]*res5[i]*res8[i]*resa[i]*resb[i]*resc[i]*resd[i]*res9[i]*res10[i]*res11[i])**(1.0/12)
'''
