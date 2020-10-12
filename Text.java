import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stemmers.NullStemmer;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stemmers.Stemmer;
import weka.core.stopwords.MultiStopwords;
import weka.core.stopwords.Null;
import weka.core.stopwords.Rainbow;
import weka.core.stopwords.RegExpFromFile;
import weka.core.stopwords.StopwordsHandler;
import weka.core.stopwords.WordsFromFile;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.core.tokenizers.CharacterNGramTokenizer;
import weka.core.tokenizers.NGramTokenizer;
import weka.core.tokenizers.Tokenizer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Text {
    public static void main(String[] args) throws Exception {
        Text tt = new Text();
        DataSource datasource = new DataSource("text.arff");
        Instances t_data = datasource.getDataSet();
        t_data.setClass(t_data.attribute("score"));
        t_data.deleteAttributeAt(6);
        t_data.deleteAttributeAt(5);
        t_data.deleteAttributeAt(4);
        t_data.deleteAttributeAt(1);
        t_data.deleteAttributeAt(0);
        System.out.println(t_data.toSummaryString());

        // config wordvector for text.arff
        StringToWordVector swFilter = new StringToWordVector();
        swFilter.setAttributeIndices("2");
        swFilter.setIDFTransform(true);
        swFilter.setTFTransform(true);
        swFilter.setDoNotOperateOnPerClassBasis(true);
        swFilter.setOutputWordCounts(true);
        // swFilter.setStemmer(new SnowballStemmer());
        // swFilter.setStopwordsHandler(new Rainbow());
        // swFilter.setTokenizer(new NGramTokenizer());
        swFilter.setWordsToKeep(100);

        ArrayList<Stemmer> stemmers = new ArrayList<Stemmer>();
        stemmers.add(new LovinsStemmer());
        stemmers.add(new SnowballStemmer());
        stemmers.add(new IteratedLovinsStemmer());
        stemmers.add(new NullStemmer());

        ArrayList<StopwordsHandler> sh = new ArrayList<StopwordsHandler>();
        sh.add(new MultiStopwords());
        sh.add(new Null());
        sh.add(new Rainbow());
        sh.add(new RegExpFromFile());
        sh.add(new WordsFromFile());

        ArrayList<Tokenizer> tokenizers = new ArrayList<Tokenizer>();
        tokenizers.add(new AlphabeticTokenizer());
        tokenizers.add(new CharacterNGramTokenizer());
        tokenizers.add(new NGramTokenizer());
        tokenizers.add(new WordTokenizer());

        try {
            FileWriter myfile = new FileWriter("Textresult.txt");
            for(int i = 0; i < stemmers.size(); i++){
                swFilter.setStemmer(stemmers.get(i));
                for(int j = 0; j < sh.size(); j++) {
                    swFilter.setStopwordsHandler(sh.get(j));
                    for(int k = 0; k < tokenizers.size(); k++) {
                        swFilter.setTokenizer(tokenizers.get(k));
                        FilteredClassifier fc = new FilteredClassifier();
                        fc.setFilter(swFilter);
                        fc.setClassifier(new J48());
                        double r = tt.doClassification(fc, t_data, new Evaluation(t_data));
                        String wr = String.valueOf(r) + "\n";
                        myfile.write(wr);
                    }
                }
            }
            myfile.close();
        } catch(IOException e) {
            System.out.println("Something Errorr");
            e.printStackTrace();
        }
        


        // create filter for text.arff
        // FilteredClassifier fc = new FilteredClassifier();
        // fc.setFilter(swFilter);

        // System.out.println("Nb");
        // fc.setClassifier(new NaiveBayes());
        // tt.doClassification(fc, t_data, new Evaluation(t_data));

        // System.out.println("J48");
        // fc.setClassifier(new J48());
        // tt.doClassification(fc, t_data, new Evaluation(t_data));

        // System.out.println("Random Forst");
        // fc.setClassifier(new RandomForest());
        // tt.doClassification(fc, t_data, new Evaluation(t_data));
    }

    public double doClassification(AbstractClassifier classifier, Instances data, Evaluation eval) throws Exception{
        classifier.buildClassifier(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        System.out.println(eval.correct());
        return eval.correct();
    }
    
}
