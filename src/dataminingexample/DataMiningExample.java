package dataminingexample;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

///////////////////////////////////////////////////////
// Observa: http://weka.wikispaces.com/Use+Weka+in+your+Java+code
///////////////////////////////////////////////////////
public class DataMiningExample {

	public static void main(String[] args) throws Exception {
		/////////////////////////////////////////////////////////////
		// ABRIR FICHERO
		String path="C:\\Universidad\\2017-2018\\2doCuatri\\SAD-sistemas de apoyo a la decisión\\Practicas\\codigoJava\\SADJava\\bin\\archivos\\heart-c.arff";
		FileReader fi = null;
		try {
			fi = new FileReader(path);
			System.out.println("Fichero abierto");
		} catch (FileNotFoundException e) {
			System.out.println("Fichero NO abierto");
			System.out.println("ERROR: Revisar path del fichero de datos:" + args[0]);
		}
		// CARGAR INSTANCIAS
		Instances data = null;
		try {
			data = new Instances(fi);
		} catch (IOException e) {
			System.out.println("ERROR: Revisar contenido del fichero de datos: " + args[0]);
		}
		// CERRAR FICHERO
		try {
			fi.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
		}

		// Falta aplicar las ?? del fichero
		// ASIGNAR EL ATRIBUTO CLASE
		data.setClassIndex(data.numAttributes() - 1);

		/////////////////////////////////////////////////////////////
		// CREAr LOS FILTROS POR LOS QUE DEBERAN PASAR LAS INSTANCIAS
		AttributeSelection filter = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(data);

		// APLICAR LOS FILTROS A LAS INSTANCIAS
		Instances newData = Filter.useFilter(data, filter);

		// CLASIFICAR SEGUN ALGORITMO NAIVEBAYES
		// ENTRENAR CLASIFICADOR
		NaiveBayes estimador = new NaiveBayes();

		// EVALUAR CLASIFICADOR MEDIANTE 10-fold CROSS-VALIADATION CON LOS DATOS
		// BARAJADOS
		Evaluation evaluator = new Evaluation(newData);
		// Random(1):the seed = 1 means "no shuffle" :-!
		evaluator.crossValidateModel(estimador, newData, 10, new Random(1)); 

		double acc = evaluator.pctCorrect();
		double inc = evaluator.pctIncorrect();
		double kappa = evaluator.kappa();
		double mae = evaluator.meanAbsoluteError();
		double rmse = evaluator.rootMeanSquaredError();
		double rae = evaluator.relativeAbsoluteError();
		double rrse = evaluator.rootRelativeSquaredError();
		double confMatrix[][] = evaluator.confusionMatrix();

		// EVALUAR CLASIFICADOR 30% DE LOS DATOS SELECCIONADOS ALEATORIAMENTE PARA EL TEST
		System.out.println("Correctly Classified Instances  " + acc);
		System.out.println("Incorrectly Classified Instances  " + inc);
		System.out.println("Kappa statistic  " + kappa);
		System.out.println("Mean absolute error  " + mae);
		System.out.println("Root mean squared error  " + rmse);
		System.out.println("Relative absolute error  " + rae);
		System.out.println("Root relative squared error  " + rrse);
		for (int i = 0; i < confMatrix.length; i++) {
			for (int j = 0; j < confMatrix[0].length; j++) {
				System.out.print(confMatrix[i][j]);
			}
			System.out.println();
		}

	}
}