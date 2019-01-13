import tensorflow as tf

PATH = "/home/newcomer/Desktop/tensorfun/"
SAVE_PATH = "/home/newcomer/Desktop/tensorfun/train" 

feature_names = [
    'Elo',
    'Pts_div_pld',
    'GfA',
    'GaA']

feature_columns = [ tf.feature_column.numeric_column(i) for i in feature_names ]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
	hidden_units = [10,10],
	n_classes = 3,
	model_dir = SAVE_PATH)

prediction_input = [[1647, 1.0, 1.25, 1.5],  # -> 0 lose
                    [1935, 2.6, 2.0, 0.4], #,  ->1 draw
                    [1869, 2.4, 2.4, 0.7],  # -> 2 win
                    [1685.09, 1.66, 1.75, 1.33]]  # -> 0 lose


# In memory predictions read
def mem_input_fn():
	def to_tensor_representation(line):
		line = tf.split(line, 4)
		return dict(zip(feature_columns, line))

	# Add data to memory
	data = tf.data.Dataset.from_tensor_slices(prediction_input)
	data = data.map(to_tensor_representation)
	iterator = data.make_one_shot_iterator()
	features = iterator.get_next()

	return features, None # Because because we want to predict

results = classifier.predict(input_fn = lambda: mem_input_fn())

for r in results:
	print r 
	idx = r["classes"][0]
	print idx, r["probabilities"][int(idx)]
