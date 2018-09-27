from glearn.policies.policy import Policy


class RandomPolicy(Policy):
    def run(self, sess, graphs, feed_map, **kwargs):
        results = {}
        output = []
        for i in range(len(feed_map["X"])):
            output.append(self.output.sample())
        results["predict"] = output
        return results
