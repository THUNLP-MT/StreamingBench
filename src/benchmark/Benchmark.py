class Benchmark:
    def __init__(self, data):
        """
        Initialize the benchmark based on the given data.
        data: data input
        """
        pass

    def eval(self, data, model, output_path, context_time):
        """
        Evaluate the model on the given data and update the data with the model responses.
        data: data input
        model: model to evaluate
        output_path: path to save the model responses
        context_time: the time before the query (not used in proactive task and open stream task)
        """
        pass