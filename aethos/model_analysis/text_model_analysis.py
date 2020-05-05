from IPython.display import HTML, display

from .model_analysis import ModelAnalysisBase


class TextModelAnalysis(ModelAnalysisBase):
    def __init__(self, model, data, model_name, **kwargs):
        """
        Class to analyze Text models through metrics and visualizations.

        Parameters
        ----------
        model : str or Model Object
            Model object or .pkl file of the objects.

        data : pd.DataFrame
            Training Data used for the model.

        model_name : str
            Name of the model for saving images and model tracking purposes

        corpus : list, optional
            Gensim LDA corpus variable. NOTE: Only for Gensim LDA

        id2word : list, optional
            Gensim LDA corpus variable. NOTE: Only for Gensim LDA
        """

        self.model = model
        self.x_train = data
        self.model_name = model_name

        # LDA dependant variables
        self.corpus = kwargs.pop("corpus", None)
        self.id2word = kwargs.pop("id2word", None)

    def view(self, original_text, model_output):
        """
        View the original text and the model output in a more user friendly format
        
        Parameters
        ----------
        original_text : str
            Column name of the original text

        model_output : str
            Column name of the model text

        Examples
        --------
        >>> m = model.LDA()
        >>> m.view('original_text_col_name', 'model_output_col_name')
        """

        results = self.x_train[[original_text, model_output]]

        display(HTML(results.to_html()))

    def view_topics(self, num_topics=10, **kwargs):
        """
        View topics from topic modelling model.
        
        Parameters
        ----------
        num_topics : int, optional
            Number of topics to view, by default 10

        Returns
        --------
        str
            String representation of topics and probabilities

        Examples
        --------
        >>> m = model.LDA()
        >>> m.view_topics()
        """

        return self.model.show_topics(num_topics=num_topics, **kwargs)

    def view_topic(self, topic_num: int, **kwargs):
        """
        View a specific topic from topic modelling model.
        
        Parameters
        ----------
        topic_num : int

        Returns
        --------
        str
            String representation of topic and probabilities

        Examples
        --------
        >>> m = model.LDA()
        >>> m.view_topic(1)
        """

        return self.model.show_topic(topicid=topic_num, **kwargs)

    def visualize_topics(self, **kwargs):  # pragma: no cover
        """
        Visualize topics using pyLDAvis.

        Parameters
        ----------
        R : int
            The number of terms to display in the barcharts of the visualization. Default is 30. Recommended to be roughly between 10 and 50.

        lambda_step : float, between 0 and 1
            Determines the interstep distance in the grid of lambda values over which to iterate when computing relevance. Default is 0.01. Recommended to be between 0.01 and 0.1.

        mds : function or {'tsne', 'mmds}
            A function that takes topic_term_dists as an input and outputs a n_topics by 2 distance matrix.
            The output approximates the distance between topics. See js_PCoA() for details on the default function.
            A string representation currently accepts pcoa (or upper case variant), mmds (or upper case variant) and tsne (or upper case variant), if sklearn package is installed for the latter two.

        n_jobs : int
            The number of cores to be used to do the computations. The regular joblib conventions are followed so -1, which is the default, will use all cores.

        plot_opts : dict, with keys ‘xlab’ and ylab
            Dictionary of plotting options, right now only used for the axis labels.

        sort_topics : bool
            Sort topics by topic proportion (percentage of tokens covered).
            Set to false to keep original topic order.

        Examples
        --------
        >>> m = model.LDA()
        >>> m.visualize_topics()
        """

        import pyLDAvis
        import pyLDAvis.gensim

        pyLDAvis.enable_notebook()

        return pyLDAvis.gensim.prepare(self.model, self.corpus, self.id2word, **kwargs)

    def coherence_score(self, col_name):
        """
        Displays the coherence score of the topic model.

        For more info on topic coherence: https://rare-technologies.com/what-is-topic-coherence/ 
        
        Parameters
        ----------
        col_name : str
            Column name that was used as input for the LDA model

        Examples
        --------
        >>> m = model.LDA()
        >>> m.coherence_score()
        """

        import gensim
        import plotly.graph_objects as go

        texts = self.x_train[col_name].tolist()

        coherence_model_lda = gensim.models.CoherenceModel(
            model=self.model, texts=texts, dictionary=self.id2word, coherence="c_v"
        )
        coherence_lda = coherence_model_lda.get_coherence()

        fig = go.Figure(
            go.Indicator(
                domain={"x": [0, 1], "y": [0, 1]},
                value=coherence_lda,
                mode="number",
                title={"text": "Coherence Score"},
            )
        )

        fig.show()

    def model_perplexity(self):
        """
        Displays the model perplexity of the topic model.

        Perplexity is a measurement of how well a probability distribution or probability model predicts a sample. It may be used to compare probability models.
        
        A low perplexity indicates the probability distribution is good at predicting the sample.

        Examples
        --------
        >>> m = model.LDA()
        >>> m.model_perplexity()
        """

        import plotly.graph_objects as go

        fig = go.Figure(
            go.Indicator(
                domain={"x": [0, 1], "y": [0, 1]},
                value=self.model.log_perplexity(self.corpus),
                mode="number",
                title={"text": "Model Perplexity"},
            )
        )

        fig.show()

        print("Note: The lower the better.")
