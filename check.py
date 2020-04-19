import json
d = json.load(open('EVAL_OUTPUT')) 
print('easy', d['total_scores']['easy']['exact'])
print('medium', d['total_scores']['medium']['exact'])
print('hard', d['total_scores']['hard']['exact'])
print('extra', d['total_scores']['extra']['exact'])
print('all', d['total_scores']['all']['exact']) # should be ~0.42
print()
# print(d['total_scores']['all'].keys())

# print(d['total_scores']['easy']) 