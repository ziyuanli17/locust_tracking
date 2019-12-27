from deepposekit import Annotator

app = Annotator(datapath='annotation_set.h5',
                dataset='images',
                skeleton='skeleton.csv',
                shuffle_colors=False,
                text_scale=0.2)
app.run()
