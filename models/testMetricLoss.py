from HeterogeneousGraph import HAN

name = "hongbin_li"

han = HAN.HAN()
features, labels, pids, rawlabels = han.loadFeature(name)

print ("res: ", features)


