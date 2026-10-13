import openvino as ov

model = ov.Core().read_model("models/tinyllama/openvino_model.xml")
ops = model.get_ordered_ops()

print(len(ops))

#a loop over ops[:15] (that means the first 15 items of the list)


#op.get_type_name()


def getfirst15(op):
    first15 = []
    for i in op[:15]:
        op.get_type_name()
        op.get_friendly_name()     
        first15.append(i)
    return first15
print(getfirst15(ops))