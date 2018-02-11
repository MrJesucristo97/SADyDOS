package dataminingexample;
 
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.instance.RemovePercentage;

///////////////////////////////////////////////////////
// Observa: http://weka.wikispaces.com/Use+Weka+in+your+Java+code
///////////////////////////////////////////////////////
public class DataMiningExample {

	public static void main(String[] args) throws Exception {
		
		/////////////////////////////////////////////////////////////
		// 1. ABRIR FICHERO(s)
		String path = args[0];
		FileReader fi = null;
		try {
			fi = new FileReader(path);
		} catch (FileNotFoundException e) {
			System.out.println("ERROR: Revisar path del fichero de datos:" + args[0]);
		}


		// 2. CARGAR INSTANCIAS
		Instances data = null;
		try {
			data = new Instances(fi);
		} catch (IOException e) {
			System.out.println("ERROR: Revisar contenido del fichero de datos: " + args[0]);
		}



		// 3. CERRAR FICHERO
		try {
			fi.close();
		} catch (IOException e) {
			System.out.println("Error al cerrar");
		}



		// 4. ASIGNAR EL ATRIBUTO CLASE
		data.setClassIndex(data.numAttributes() - 1);


		/**
        ///////////////////////////- OPCIONAL -//////////////////////////////////
        // 5. CREAR LOS FILTROS POR LOS QUE DEBERAN PASAR LAS INSTANCIAS
        AttributeSelection filter = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);

        // 6. APLICAR LOS FILTROS A LAS INSTANCIAS
        data = Filter.useFilter(data, filter);
        ///////////////////////////////////////////////////////////////

		

		//Filtro remove percentage (Para hold out)
		RemovePercentage filter = new RemovePercentage();
		filter.setPercentage(70.0);
		filter.setInputFormat(data);
		Instances dataEntrenamiento = Filter.useFilter(data, filter);
		filter.setInvertSelection(true);
		filter.setPercentage(30.0);
		Instances dataTest = Filter.useFilter(data, filter); **/
		
		
		
		

		// 7. ELEGIR ALGORITMO PARA CLASIFICAR

		// NAIVEBAYES
		//NaiveBayes clasificador = new NaiveBayes(); // entrenar clasificador

		// ZEROR
		//ZeroR clasificador = new ZeroR();

		// ONER
		//OneR clasificador = new OneR();
		
		// IBk
		//IBk clasificador = new IBk();

		//J48
		J48 clasificador = new J48(); 
		String[] options = new String[1];
		options[0] = "-U"; //unpruned tree
		clasificador.setOptions(options);
		
		
	
		// 8. ELEGIR ESQUEMA DE EVALUACIÓN (Test options)
		Evaluation evaluator;
		
			

		//NO-HONESTA (use training set) -- Supplied test set
		clasificador.buildClassifier(data);   // construir clasificador

		evaluator = new Evaluation(data); //datos para entrenar

		Instances test = data; //datos para test

		evaluator.evaluateModel(clasificador, test);


		//HOLD-OUT (percentage split, ejemplo 70% entrenamiento, 30% evaluación)
		double percent = 70.0; 
		
		data.randomize(new java.util.Random(1)); //aleatoriedad de selección de datos

	    int tamanoEntrenamiento = (int) Math.round(data.numInstances() * (percent / 100)); 
		int tamanoTest = data.numInstances() - tamanoEntrenamiento; 

		Instances datosEntrenamiento = new Instances(data, 0, tamanoEntrenamiento); 
		Instances datosTest = new Instances(data, tamanoEntrenamiento, tamanoTest); 

		clasificador.buildClassifier(dataEntrenamiento);   // construir clasificador

		evaluator = new Evaluation(dataEntrenamiento);
		evaluator.evaluateModel(clasificador, dataTest);


		
		// 10-fold CROSS-VALIDATION CON LOS DATOS
		// BARAJADOS
		// Random(1):the seed = 1 means "no shuffle" :-!
		evaluator = new Evaluation(data); //datos para entrenar
		evaluator.crossValidateModel(clasificador, data, 10, new Random(1)); 


		
		//LEAVE ONE OUT (K=nº instancias)
		evaluator = new Evaluation(data); //datos para entrenar
		evaluator.crossValidateModel(clasificador, data, data.numInstances(), new Random(1)); 



		// OUTPUT
		System.out.print(evaluator.toSummaryString());
		System.out.print(evaluator.toMatrixString());

	}
}