package org.naomod.yakindu2satecharts;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Map;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.eclipse.m2m.atl.emftvm.EmftvmFactory;
import org.eclipse.m2m.atl.emftvm.ExecEnv;
import org.eclipse.m2m.atl.emftvm.Metamodel;
import org.eclipse.m2m.atl.emftvm.Model;
import org.eclipse.m2m.atl.emftvm.impl.resource.EMFTVMResourceFactoryImpl;
import org.eclipse.m2m.atl.emftvm.util.DefaultModuleResolver;
import org.eclipse.m2m.atl.emftvm.util.ModuleResolver;
import org.eclipse.m2m.atl.emftvm.util.TimingData;

public class RunTransformations {
	
	public static String here = new File(".").getAbsolutePath();
	
	public static URI resourceURI(String relativePath) {
		return URI.createFileURI(here + relativePath);
	}
	
	public static void main(String[] args) throws IOException {
		
		Map<String, Object> map = Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap();
		map.put("xmi", new XMIResourceFactoryImpl());
		map.put("ecore", new EcoreResourceFactoryImpl());
		map.put("emftvm", new EMFTVMResourceFactoryImpl());

		ExecEnv env = EmftvmFactory.eINSTANCE.createExecEnv();
		ResourceSet rs = new ResourceSetImpl();
		
		EPackage YakinduPkg = (EPackage) rs.getResource(resourceURI("/../resources/metamodels/yakindu.ecore"), true)
				.getContents().get(0);
		EPackage.Registry.INSTANCE.put(YakinduPkg.getNsURI(), YakinduPkg);
		EPackage StateChartsPkg = (EPackage) rs.getResource(resourceURI("/../resources/metamodels/statecharts.ecore"), true)
				.getContents().get(0);
		EPackage.Registry.INSTANCE.put(StateChartsPkg.getNsURI(), StateChartsPkg);

		// Load metamodels
		Metamodel yakinduMetaModel = EmftvmFactory.eINSTANCE.createMetamodel();
		yakinduMetaModel.setResource(rs.getResource(resourceURI("/../resources/metamodels/yakindu.ecore"), true));
		env.registerMetaModel("Yakindu", yakinduMetaModel);
		
		Metamodel stateChartsMetaModel = EmftvmFactory.eINSTANCE.createMetamodel();
		stateChartsMetaModel.setResource(rs.getResource(resourceURI("/../resources/metamodels/statecharts.ecore"), true));
		env.registerMetaModel("Statecharts", stateChartsMetaModel);
		

		File InDir = new File("../resources/models/yakindu_input");

		int i = 0;
		String basePath = "C:/Users/James/Projects/Eclipse/transformations-emf-views/Java";
		for (File file : InDir.listFiles()) {
			
			if (file.getName().toString().equals(".gitkeep")) {
				continue;
			}
			
			

			System.out.println("File: " + file.getAbsolutePath());
			
			// Load models
			Model inModel = EmftvmFactory.eINSTANCE.createModel();
			inModel.setResource(rs.getResource(resourceURI("/../resources/models/yakindu_input/" + file.getName()), true));
			env.registerInputModel("IN", inModel);
			
			String relativePathTrace = "../resources/models/statecharts_output/trace_" + file.getName();
			String fullPathTrace = basePath + "/" + relativePathTrace;
			URI uriTrace = URI.createFileURI(fullPathTrace);
			
			Model traceOutModel = EmftvmFactory.eINSTANCE.createModel();
			traceOutModel.setResource(rs.createResource(uriTrace));
			env.registerOutputModel("trace", traceOutModel);
			
			String relativePathOut = "../resources/models/statecharts_output/" + file.getName();
			String fullPathOut = basePath + "/" + relativePathOut;
			URI uriOut = URI.createFileURI(fullPathOut);
			
			Model outModel = EmftvmFactory.eINSTANCE.createModel();
			outModel.setResource(rs.createResource(uriOut));
			env.registerOutputModel("OUT", outModel);
			
			// Load and run module
			ModuleResolver mr = new DefaultModuleResolver("./../Yakindu2StateCharts/emftvm/",
					rs);
			TimingData td = new TimingData();
			env.loadModule(mr, "Yakindu2StateCharts");
			td.finishLoading();
			env.run(td);
			td.finish();
			
			// Save models
			inModel.getResource().save(Collections.emptyMap());
			traceOutModel.getResource().save(Collections.emptyMap());
			outModel.getResource().save(Collections.emptyMap());

		}
	}
}
