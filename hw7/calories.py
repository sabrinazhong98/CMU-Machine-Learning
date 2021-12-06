# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 23:52:16 2021
https://www.youtube.com/watch?v=7AYuiPvbDew
@author: zhong
"""

gender  = str(input('female or male: '))
bodyweight = float(input('bodyweight in kg: '))
bodyweight_lbs = 2.20462*bodyweight
height = int(input('height in cm: '))
age = int(input('age: '))
workout = float(input('workout index from 1 to 2: '))
calories = float(input('type in calories deficit: from - to +: '))
bodyfat = float(input('type in body fat 0 to 1: '))
leanbody = bodyweight_lbs - (bodyweight_lbs*bodyfat)

if gender == 'male':
    bmr = 66+(13.7*bodyweight) + (5*height)-(6.8*age)
else:
    bmr = 665+(9.6*bodyweight) + (1.7*height)-(4.7*age)

daily_energy_expenditure = bmr * workout
expected_calories = daily_energy_expenditure + calories*daily_energy_expenditure

#middle carbs day
mid_carbs = 1.25*bodyweight_lbs
mid_pro = 1.2*leanbody
mid_fat_low = (expected_calories - (mid_carbs+mid_pro)*4)/9
mid_fat_high= (expected_calories - (mid_carbs+leanbody)*4)/9

print('total calories: ', expected_calories)
print('mid day carbs: {0:.2f}g protein: {1:.2f} to {2:.2f}g fat: {3:.2f} to {4:.2f}g'.format(mid_carbs, leanbody,mid_pro,mid_fat_high, mid_fat_low))

#high carbs day
high_carbs = 1.75*mid_carbs
high_fat_low = 0.5*mid_fat_low
high_fat_high = 0.5*mid_fat_high

print('high day carbs: {0:.2f} protein: {1:.2f} to {2:.2f} fat: {3:.2f} to {4:.2f}g'.format(high_carbs, leanbody,mid_pro,high_fat_high, high_fat_low))

#low carbs day
low_carbs = 0.25*mid_carbs
low_fat_low = 1.5*mid_fat_low
low_fat_high = 1.5*mid_fat_high

print('low day carbs: {0:.2f} protein: {1:.2f} to {2:.2f} fat: {3:.2f} to {4:.2f}g'.format(low_carbs, leanbody,mid_pro,low_fat_high, low_fat_low))






