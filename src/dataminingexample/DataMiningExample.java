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
		// 1. ABRIR FICHERO(s)
		String path="C:\\Users\\Pauladj\\Documents\\GitHub\\SADyDOS\\src\\archivos\\heart-c.arff";
		FileReader fi = null;
		try {
			fi = new FileReader(path);
			System.out.println("Fichero abierto");
		} catch (FileNotFoundException e) {
			System.out.println("Fichero NO abierto");
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

		 **/

		// 7. ELEGIR ALGORITMO PARA CLASIFICAR

		// NAIVEBAYES
		NaiveBayes estimador = new NaiveBayes(); // entrenar clasificador
		estimador.buildClassifier(data);   // construir clasificador

		

		// 8. ELEGIR ESQUEMA DE EVALUACIÓN (Test options)
		Evaluation evaluator;

		
		//NO-HONESTA (use training set) -- Supplied test set
		evaluator = new Evaluation(data); //datos para entrenar
		
		Instances test = data; //datos para test
		
		evaluator.evaluateModel(estimador, test);
		
		
		//-----------------------no me da como en weka
		//HOLD-OUT (percentage split, ejemplo 70% entrenamiento, 30% evaluación)
		double percent = 70.0; 
		int tamanoEntrenamiento = (int) Math.round(data.numInstances() * percent / 100); 
		int tamanoTest = data.numInstances() - tamanoEntrenamiento; 
		
		Instances datosEntrenamiento = new Instances(data, 0, tamanoEntrenamiento); 
		Instances datosTest = new Instances(data, tamanoEntrenamiento, tamanoTest); 
		
		evaluator = new Evaluation(datosEntrenamiento);
		evaluator.evaluateModel(estimador, datosTest);
		
		
		// 10-fold CROSS-VALIDATION CON LOS DATOS
		// BARAJADOS
		// Random(1):the seed = 1 means "no shuffle" :-!
		evaluator = new Evaluation(data); //datos para entrenar
		evaluator.crossValidateModel(estimador, data, 10, new Random(1)); 
		
		
		//LEAVE ONE OUT (K=nº instancias)
		evaluator = new Evaluation(data); //datos para entrenar
		evaluator.crossValidateModel(estimador, data, data.numInstances(), new Random(1)); 
		
		
		// OUTPUT
		System.out.print(evaluator.toSummaryString());
		System.out.print(evaluator.toMatrixString());

	}
}