train_instances = 50000
with open('train.txt', 'w') as f:
    for i in range(train_instances):
        f.write('{}\n'.format(i))

test_instances = 2000
with open('eval.txt', 'w') as f:
    for i in range(train_instances, train_instances+test_instances):
        f.write('{}\n'.format(i))
