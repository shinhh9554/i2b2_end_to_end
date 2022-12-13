import os
import codecs
import numpy as np


def to_joint(ner_filename, re_filename, joint_filename):
    ner_instances = []
    re_instances = {}

    with codecs.open(ner_filename, encoding="utf-8") as f:
        instance = []
        for line in f:
            line = line.strip()
            item = line.split("\t")

            if len(item) == 3:
                instance.append(item)
            elif instance:
                ner_instances.append(instance)
                instance = []

    with codecs.open(re_filename, encoding="utf-8") as f:
        instance = []
        for line in f:
            line = line.strip()
            item = line.split("\t")

            if len(item) == 5:
                instance.append(item)
            elif line:
                if line == "NONE":
                    instance = []
                    continue
                key_instance = []
                entity1 = []
                entity2 = []
                start1 = True
                start2 = True
                for i, item in enumerate(instance):
                    key_instance.append([item[0], item[1], item[-1]])
                    if item[2] == "0":
                        if start1:
                            entity1.append(i)
                    elif item[2] == "1":
                        entity1.append(i)

                    if item[3] == "0":
                        if start2:
                            entity2.append(i)
                    elif item[3] == "1":
                        entity2.append(i)

                if len(entity1) == 1:
                    entity1.append(len(instance))
                if len(entity2) == 1:
                    entity2.append(len(instance))

                pairs = [entity1, entity2]

                if str(key_instance) not in re_instances:
                    re_instances[str(key_instance)] = {"list":key_instance, line: pairs}
                else:
                    if line not in re_instances[str(key_instance)]:
                        re_instances[str(key_instance)][line] = pairs
                instance = []

    instances = []
    for ner_instance in ner_instances:
        if str(ner_instance) not in re_instances:
            instances.append(ner_instance)

    for _, vocab in re_instances.items():
        instance = vocab["list"]
        for rel_type, pair in vocab.items():
            if rel_type == "list":
                continue
            entity1 = pair[0]
            entity2 = pair[1]

            try:
                if instance[entity1[0]][-1].count("-") == 2 or instance[entity2[0]][-1].count("-") == 2:
                    continue

                new_instance = []
                for i, item in enumerate(instance):
                    if entity1[0] <= i < entity1[-1]:
                        item[-1] += "-%s1" % rel_type
                    if entity2[0] <= i < entity2[-1]:
                        item[-1] += "-%s2" % rel_type
                    new_instance.append(item)
                if new_instance:
                    instance = new_instance
            except Exception:
                print("ERR")
                continue
        instances.append(instance)

    np.random.shuffle(instances)

    with codecs.open(joint_filename, "w", encoding="utf-8") as f:
        for instance in instances:
            for item in instance:
                f.write("%s\t%s\t%s\n" % tuple(item))
            f.write("\n")


if __name__ == '__main__':
    to_joint("I2B2-2/valid.txt", "I2B2/valid.txt", "joint/valid.txt")


