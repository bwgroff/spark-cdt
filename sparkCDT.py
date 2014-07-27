from pyspark import SparkContext, SparkConf
import numpy as np
import time
from scipy.spatial.distance import cdist
import json

conf = SparkConf().setAppName("SparkCDT").setMaster("local")
sc = SparkContext(conf=conf)


def child_name(this_node_name, child_number, max_children):
    base = this_node_name * max_children + 1
    return base + child_number


def get_centroids(distdata):
    sums = distdata.reduceByKey(lambda a, b: a + b).collect() # combine by key instead??
    counts = distdata.countByKey()
    centroids = {}
    for sumtroid in sums:
        centroids[sumtroid[0]] = sumtroid[1] / counts[sumtroid[0]]
    return centroids


def nearest_centroid(x, centroids):
    dists = []
    for centroid in centroids:
        dist = (x - centroids[centroid])**2
        dists.append(np.sum(dist))
    dists = np.array(dists)
    return np.argmin(dists)


def stop_conditions(data, node, granularity, purity_tol, max_depth, max_breadth):
#     print "Evaluating stop conditions"
    class_counts = data.countByKey()
    num_rows = sum(class_counts.values())
    too_few_pts = num_rows <= granularity
    too_deep = node > max_breadth ** max_depth
    largest_class = max(class_counts, key=lambda x: class_counts[x])
    purity = (float(class_counts[largest_class]) / num_rows)
    pure_enough = purity > purity_tol
    stop = pure_enough or too_few_pts or too_deep
    return stop, purity, largest_class, num_rows


def CDT(data, this_node_name, this_centroid, granularity, purity_tol, max_depth, max_breadth, current_depth):
    depth_string = "]" + "  ]" * current_depth + " "
    print depth_string + "node: " + str(this_node_name) + " count: " + str(data.count())
    current_depth += 1
    node = {'name': this_node_name, 'centroid': [float(x) for x in this_centroid]}
#     t0 = time.time()
    stop, purity, dominant_class, num_rows = stop_conditions(data, this_node_name, granularity,
                                                   purity_tol, max_depth, max_breadth)
#     t1 = time.time()
#     print "stop condition eval time: " + str(t1-t0)
    if not stop:
        children = []
#         t2 = time.time()
        centroids = get_centroids(data)
#         t3 = time.time()
#         print "get_centroids eval time: " + str(t3-t2)
        for l, centroid in enumerate(centroids):
#             t4 = time.time()
            label_filtered_data = data.filter(lambda x: nearest_centroid(x[1], centroids) == l)
#             t5 = time.time()
#             print "data filter eval time: " +str(t5-t4)
            child = CDT(label_filtered_data, child_name(this_node_name, l, max_breadth),
                        centroids[centroid], granularity, purity_tol, max_depth, max_breadth, current_depth)
            children.append(child)
        node['children'] = children
    else:
        print ']' + "::|" * current_depth + " purity: " + str(purity)
        node['purity'] = purity
        node['dominant class'] = dominant_class
    current_depth -= 1
    return node


def predict(cdt, pt):
    finished = False
    current_node = cdt
    while not finished:
        centroids = np.array([child['centroid'] for child in current_node['children']])
        best_centroid = np.argmin(cdist([pt], centroids), axis=1)
        current_node = current_node['children'][best_centroid[0]]
        finished = 'dominant class' in current_node
    return current_node['dominant class']


def save_model(self, filename, indent=0):
    json.dump(self.bread, open(filename, "w"), indent=indent)

if __name__ == '__main__':
    total_pts = 10000
    data = np.reshape(np.random.uniform(0, 1, total_pts * 2), (total_pts, 2))
    data_tagged = []
    for d in data:
        data_tagged.append((int((d[0] - d[1]) * (d[0] + d[1] - 1) > 0), d))
    distdata = sc.parallelize(data_tagged)

    t0 = time.time()
    cdt = CDT(distdata, 0, [0.0, 0.0], 10, 0.999, 20, 2, 0)
    json.dump(cdt, open("cdt.json", "w"), indent=2)
    print "Model exported to file cdt.json"
    print "Elapsed time: " + str(time.time() - t0)