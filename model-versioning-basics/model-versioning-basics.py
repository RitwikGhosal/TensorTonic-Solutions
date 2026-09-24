def promote_model(models):
    best = models[0]
    for model in models[1:]:
        better_accuracy = model["accuracy"] > best["accuracy"]
        equal_accuracy = model["accuracy"] == best["accuracy"]
        better_latency = model["latency"] < best["latency"]
        equal_latency = model["latency"] == best["latency"]
        newer = model["timestamp"] > best["timestamp"]
        if better_accuracy or (equal_accuracy and better_latency) or (equal_accuracy and equal_latency and newer):
            best = model
    return best["name"]