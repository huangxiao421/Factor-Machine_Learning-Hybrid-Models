import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import SparsePCA

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Runtime track
START_TIME = datetime.now()

# Plot style
plt.style.use('seaborn')
plt.gcf().subplots_adjust(bottom=0.15)



# T = initial train size + validation size
START_T = 40
VALIDATION_SIZE = 10
TUNE = True
SAVE = True

# Data directories
directory_import = {
    "key_figures" : "C:/Users/richi/Documents/Econometrie en Operationele Research/Bachelor 3/Blok 5 (Thesis)/Datasets (processed)/key_figures_preprocessed.xlsx",
    "house_price" : "C:/Users/richi/Documents/Econometrie en Operationele Research/Bachelor 3/Blok 5 (Thesis)/Datasets (processed)/house_price_preprocessed.xlsx",
    }   

# Data directories non-gaussian
directory_import_ng = {
    "key_figures" : "C:/Users/richi/Documents/Econometrie en Operationele Research/Bachelor 3/Blok 5 (Thesis)/Datasets (processed)/key_figures_preprocessed_NG.xlsx",
    "house_price" : "C:/Users/richi/Documents/Econometrie en Operationele Research/Bachelor 3/Blok 5 (Thesis)/Datasets (processed)/house_price_preprocessed_NG.xlsx",
    }   

directory_export = "C:/Users/richi/Documents/Econometrie en Operationele Research/Bachelor 3/Blok 5 (Thesis)/Results"



def directory_dict_data_extract(directories):
    start_date = pd.to_datetime("2001-1-1").date()
    
    # Gets all excel tables and puts them in dict, using keys of data_dictionary
    data = dict()
    for key, value in directories.items():         
        data[key] = pd.read_excel(value, sheet_name = 0)
        # Parse datetime to date 
        data[key]["Datum"] = data[key]["Datum"].apply(lambda x: x.date())
        
        data[key] = data[key].set_index(['Datum'])
        data[key] = data[key].loc[start_date:]
        
    return data

data = directory_dict_data_extract(directory_import)
data_ng = directory_dict_data_extract(directory_import_ng)



def data_preprocess(data, t, scale):
    """
    Splits and scales data (scaling only if input data)
    
    Parameters
    ----------
    data :  DF / Series (error if other type)
            contains input/target data (which will be split)
    t :     int
            current time (t of test data)
    scale : boolean
            Scale = True, if data = input 

    Returns
    -------
    train, validation, test, current_date (at t)
    
    """
    # Split data (reshape s.t. single sample is accepted in scaling), series/df require different code
    if isinstance(data, pd.Series):
        train       = data[0:t - VALIDATION_SIZE].values
        validation  = data[t - VALIDATION_SIZE : t].values
        test        = data[t].reshape(1, -1)
    elif isinstance(data, pd.DataFrame):
        train       = data.iloc[0:t - VALIDATION_SIZE, :].values
        validation  = data.iloc[t - VALIDATION_SIZE : t, :].values     
        test        = data.iloc[t, :].values.reshape(1, -1)
    else:
        raise Exception("Wrong data type => should be Series or DF")
    
    # Scale data (input only)
    if scale == True:
        scaler = StandardScaler()
        scaler.fit(train)
        train       = scaler.transform(train)
        validation  = scaler.transform(validation)
        test        = scaler.transform(test)
        
    # Get current date from index
    current_date = data.index[t]
    
    return train, validation, test, current_date 



def pca(k, train_input, train_target, test_input, test_target, model):
    """
    Executes PCA by creating/estimating components based on train data.
    Then tests performance on validation/test data.
    
    Parameters
    ----------
    k : int = amount of PC's
    model : model-type that's used to in conjunction with the components
    
    Returns
    -------
    prediction : Validation/Test forecasts (based on test_input/test_target)
    mse : error of predictions
    
    """
    # Create PC's from training data
    pca = PCA(k)
    pca.fit(train_input)
    eigenvectors = pca.components_.T
    pc_train = np.dot(train_input, eigenvectors)
    
    # Least Squares to fit train target
    model.fit(pc_train, train_target)
    
    # Test / Validate
    pc_test = np.dot(test_input, eigenvectors)
    prediction = model.predict(pc_test)
    mse = mean_squared_error(test_target, prediction)
    
    return prediction, mse



def ica(k, train_input, train_target, test_input, test_target, model):
    """
    Executes ICA by creating/estimating components based on train data.
    Then tests performance on validation/test data.
    
    Parameters
    ----------
    k : int = amount of IC's
    model : model-type that's used to in conjunction with the components
    
    Returns
    -------
    prediction : Validation/Test forecasts (based on test_input/test_target)
    mse : error of predictions
    
    """
    pca = PCA(k)
    pca.fit(train_input)    
    eigen_vectors = pca.components_.T
    principal_components = np.dot(train_input, eigen_vectors) 
    
    # Create PC's from training data
    ica = FastICA(k, max_iter = 200)
    ica.fit(principal_components)
    eigen_vectors = ica.components_.T
    ic_train = np.dot(train_input, eigen_vectors)
    
    # Least Squares to fit train target
    model.fit(ic_train, train_target)
    
    # Test / Validate
    ic_test = np.dot(test_input, eigen_vectors)
    prediction = model.predict(ic_test)
    mse = mean_squared_error(test_target, prediction)
    
    return prediction, mse



def spca(k, train_input, train_target, test_input, test_target, model):
    """
    Executes SPCA by creating/estimating components based on train data.
    Then tests performance on validation/test data.
    
    Parameters
    ----------
    k : int = amount of SPC's
    model : model-type that's used to in conjunction with the components
    
    Returns
    -------
    prediction : Validation/Test forecasts (based on test_input/test_target)
    mse : error of predictions
    
    """
    # Create PC's from training data
    spca = SparsePCA(k)
    spca.fit(train_input)
    eigenvectors = spca.components_.T
    spc_train = np.dot(train_input, eigenvectors)
    
    # Least Squares to fit train target
    model.fit(spc_train, train_target)
    
    # Test / Validate
    spc_test = np.dot(test_input, eigenvectors)
    prediction = model.predict(spc_test)
    mse = mean_squared_error(test_target, prediction)
    
    return prediction, mse



def main(data, tune, variation): 
    print(f"\n----------- progress {variation.upper()}, Single tune")
    if tune == True:
        # Hyperparameter tuning
        hyperparameters = dict()
        tune_results = dict()
        
        # Tune every target separately (w/ grid search)
        for column in data["house_price"]:
            # Grid search   
            input_data = data["key_figures"]
            target_data = data["house_price"][column]
            all_combinations, best_combination = xgboost_tuning(input_data, target_data, variation, START_T)
            
            # append to dictionary 
            tune_results[column] = all_combinations
            hyperparameters[column] = best_combination
        print(f"finished tuning: \n{hyperparameters}  \n{datetime.now() - START_TIME}\n")
    else:
        hyperparameters = None
        tune_results = None
        print("No tuning")
        
    # Initialise for final output
    output = []
    parameters = []
    input_data = data["key_figures"]
    
    # t is where training + validation ends, thus testing = t + 1
    for t in range(START_T, len(input_data)):
        # Output for current t
        output_t = [] 
        parameters_t = []
        
        for column in data["house_price"]:
            # Specify model & (tuned) parameters
            if tune == True:
                model = XGBRegressor(eta = hyperparameters[column]["eta"], max_depth = hyperparameters[column]["max_depth"])
            else:
                model = XGBRegressor()
                
            # Get single target series + input DF
            
            target_data = data["house_price"][column]
            
            # Get data split (t + 1 for target to forecast 1-step ahead)
            train_input, validation_input, test_input, current_date = data_preprocess(input_data, t, scale = True)
            train_target, validation_target, test_target, _ = data_preprocess(target_data, t + 1, scale = False)
            
            # Delete first element train_target, because we forecast 1-step ahead (note expanding window, otherwise we get length "input length + 1")
            train_target = np.delete(train_target, 0)
            
            # Find best k, through trial & error (MSE minimization)     
            mse_list = []    
            k_list = []
            
            for k in range(2, 22, 2):
                # PCR for optimal k 
                if variation == "pca":
                    _ , mse = pca(k, train_input, train_target, validation_input, validation_target, model)
                elif variation == "spca":
                    _ , mse = spca(k, train_input, train_target, validation_input, validation_target, model)
                elif variation == "ica":
                    break # ICA does not tune amount of components, but uses amount of characteristics 
                else:
                    raise Exception("component variation wrongly specified \nShould be in ['pca', 'spca', 'ica']") 
                
                mse_list.append(mse)
                k_list.append(k)
            
            if variation == "ica":
                k_optimal = train_input.shape[1] # amount of input characteristics
            else:
                k_optimal = k_list[mse_list.index(min(mse_list))] # pca/spca use k with best mse

            # PCR for optimal k  
            if variation == "pca":
                prediction , mse = pca(k_optimal, train_input, train_target, test_input, test_target, model)
            elif variation == "spca":
                prediction , mse = spca(k_optimal, train_input, train_target, test_input, test_target, model)
            elif variation == "ica":
                prediction , mse = ica(k_optimal, train_input, train_target, test_input, test_target, model)
            else:
                raise Exception("component variation wrongly specified \nShould be in ['pca', 'spca', 'ica']")                 
            
            # Save output
            output_t.append(current_date)
            output_t.append(test_target.item())
            output_t.append(prediction.item())
        
            # Save hyperparameters
            parameters_t.append(current_date)
            parameters_t.append(hyperparameters[column]["eta"])
            parameters_t.append( hyperparameters[column]["max_depth"])
            parameters_t.append(k_optimal)
                
        
        print(f"t = {t} - {current_date}")
        # Save output (current t)
        output.append(output_t)
        parameters.append(parameters_t)
        
    # Parse output from nested lists to DF
    output = pd.DataFrame(output, columns = ["Date", "price_index_ACTUAL", "price_index_FORECAST", "", "avg_selling_price_ACTUAL", "avg_selling_price_FORECAST"])
    output.drop(output.columns[3], axis = 1, inplace = True)
    
    parameters = pd.DataFrame(parameters, columns = ["Date", "price_index_ETA", "price_index_MAX_DEPTH", "price_index_K", "", "avg_selling_price_ETA", "avg_selling_price_MAX_DEPTH", "avg_selling_price_K"])
    parameters.drop(parameters.columns[4], axis = 1, inplace = True)
        
    print(f"----------- finished {variation.upper()} - {datetime.now() - START_TIME}")
    return output, tune_results, parameters
        


def main_tune_every_t(data, tune, variation): 
    print(f"\n----------- progress {variation.upper()}, Tune every T")   
    # Initialise for final output
    output = []
    parameters = []
    
    # t is where training + validation ends, thus testing = t + 1
    for t in range(START_T, len(data["key_figures"])):
        # Output for current t
        output_t = [] 
        parameters_t = []
        
        for column in data["house_price"]:
            # Specify model & (tuned) parameters
            if tune == True:
                # Grid search   
                input_data = data["key_figures"]
                target_data = data["house_price"][column]
                all_combinations, best_combination = xgboost_tuning(input_data, target_data, variation, t)
                
                # append to dictionary 
                hyperparameters = dict()
                hyperparameters[column] = best_combination
                
                # Specify model
                model = XGBRegressor(eta = hyperparameters[column]["eta"], max_depth = hyperparameters[column]["max_depth"])                     
            else:
                model = XGBRegressor()
            
                
            # Get single target series + input DF
            target_data = data["house_price"][column]
            
            # Get data split (t + 1 for target to forecast 1-step ahead)
            train_input, validation_input, test_input, current_date = data_preprocess(input_data, t, scale = True)
            train_target, validation_target, test_target, _ = data_preprocess(target_data, t + 1, scale = False)
            
            # Delete first element train_target, because we forecast 1-step ahead (note expanding window, otherwise we get length "input length + 1")
            train_target = np.delete(train_target, 0)
            
            # Find best k, through trial & error (MSE minimization)     
            mse_list = []    
            k_list = []
            
            for k in range(2, 22, 2):
                # PCR for optimal k 
                if variation == "pca":
                    _ , mse = pca(k, train_input, train_target, validation_input, validation_target, model)
                elif variation == "spca":
                    _ , mse = spca(k, train_input, train_target, validation_input, validation_target, model)
                elif variation == "ica":
                    break # ICA does not tune amount of components, but uses amount of characteristics 
                else:
                    raise Exception("component variation wrongly specified \nShould be in ['pca', 'spca', 'ica']") 
                
                mse_list.append(mse)
                k_list.append(k)
            
            if variation == "ica":
                k_optimal = train_input.shape[1] # amount of input characteristics
            else:
                k_optimal = k_list[mse_list.index(min(mse_list))] # pca/spca use k with best mse

            # PCR for optimal k  
            if variation == "pca":
                prediction , mse = pca(k_optimal, train_input, train_target, test_input, test_target, model)
            elif variation == "spca":
                prediction , mse = spca(k_optimal, train_input, train_target, test_input, test_target, model)
            elif variation == "ica":
                prediction , mse = ica(k_optimal, train_input, train_target, test_input, test_target, model)
            else:
                raise Exception("component variation wrongly specified \nShould be in ['pca', 'spca', 'ica']")                 
            
            # Save output
            output_t.append(current_date)
            output_t.append(test_target.item())
            output_t.append(prediction.item())
            
            # Save hyperparameters
            parameters_t.append(current_date)
            parameters_t.append(hyperparameters[column]["eta"])
            parameters_t.append( hyperparameters[column]["max_depth"])
            parameters_t.append(k_optimal)
            
        print(f"t = {t} - {current_date}")
        # Save output (current t)
        output.append(output_t)
        parameters.append(parameters_t)
        
    # Parse output from nested lists to DF
    output = pd.DataFrame(output, columns = ["Date", "price_index_ACTUAL", "price_index_FORECAST", "", "avg_selling_price_ACTUAL", "avg_selling_price_FORECAST"])
    output.drop(output.columns[3], axis = 1, inplace = True)
    
    parameters = pd.DataFrame(parameters, columns = ["Date", "price_index_ETA", "price_index_MAX_DEPTH", "price_index_K", "", "avg_selling_price_ETA", "avg_selling_price_MAX_DEPTH", "avg_selling_price_K"])
    parameters.drop(parameters.columns[4], axis = 1, inplace = True)
    
    print(f"----------- finished {variation.upper()} - {datetime.now() - START_TIME}")
    return output, parameters



def xgboost_tuning(input_data, target_data, variation, t):
    # specify hyperparameters 
    eta_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 1.25, 1.5, 2]
    max_depth_list = [1, 2, 3, 5, 6, 10]
    # Get all combinations
    hyperparameters = list(itertools.product(*[eta_list, max_depth_list]))
    
    # Get data split (t + 1 for target to forecast 1-step ahead)
    train_input, validation_input, test_input, current_date = data_preprocess(input_data, t, scale = True)
    train_target, validation_target, test_target, _ = data_preprocess(target_data, t + 1, scale = False)
    
    # Delete first element train_target, because we forecast 1-step ahead (note expanding window, otherwise we get length "input length + 1")
    train_target = np.delete(train_target, 0)
    
    mse_list = []    
    for combination in hyperparameters:
        eta = combination[0]
        max_depth = combination[1]
        model = XGBRegressor(eta = eta, max_depth = max_depth)
        
        # PCR for optimal k 
        if variation == "pca":
            _ , mse = pca(10, train_input, train_target, validation_input, validation_target, model)
        elif variation == "spca":
            _ , mse = spca(10, train_input, train_target, validation_input, validation_target, model)
        elif variation == "ica":
            _ , mse = ica(26, train_input, train_target, validation_input, validation_target, model)
        else:
            raise Exception("component variation wrongly specified \nShould be in ['pca', 'spca', 'ica']") 
             
        mse_list.append([mse])
            
    # Get combination hyperparameters of lowest MSE
    best_combination = hyperparameters[mse_list.index(min(mse_list))]
    best_combination = {"eta" : best_combination[0], "max_depth" : best_combination[1]}
    
    # Create table with corresponding tune results
    tune_results = np.asarray(mse_list).reshape(len(eta_list), len(max_depth_list))
    tune_results = pd.DataFrame(tune_results, index = eta_list, columns = max_depth_list )    

    return tune_results, best_combination    



def make_individual_plots(results, model, variation):
    # price index
    plt.plot(results["Date"], results["price_index_ACTUAL"], label = "Actual")
    plt.plot(results["Date"], results["price_index_FORECAST"], label = "Forecast")
    
    plt.xticks(rotation = 45)    
    plt.title(f"Price Index - {model} ({variation})")
    plt.ylabel('Growth rate')
    plt.legend() 
    plt.show()
    
    # avg selling price
    plt.plot(results["Date"], results["avg_selling_price_ACTUAL"], label = "Actual")
    plt.plot(results["Date"], results["avg_selling_price_FORECAST"], label = "Forecast")
    
    plt.xticks(rotation = 45)    
    plt.title(f"Average Selling Price - {model} ({variation})")
    plt.ylabel('Growth rate')
    plt.legend() 
    plt.show()



def make_joint_plot(joint_results, tet):
    # X-axis uses dates
    x = joint_results["pca"]["Date"]

    # For both targets, make joint plot
    for target in ["price_index", "avg_selling_price"]:
        # Get forecasts/actuals
        y_actual = joint_results["pca"][f"{target}_ACTUAL"]
        y_pca = joint_results["pca"][f"{target}_FORECAST"]
        y_spca = joint_results["spca"][f"{target}_FORECAST"]
        y_ica = joint_results["ica"][f"{target}_FORECAST"]
        
        # Make plots
        plt.plot(x, y_actual, label = "Actual", color = "black")
        plt.plot(x, y_pca, label = "Forecast PCA")
        plt.plot(x, y_spca, label = "Forecast SPCA")
        plt.plot(x, y_ica, label = "Forecast ICA")
        
        # Specifications
        plt.xticks(rotation = 45)    
        plt.title(f"{target.upper()} - XGBOOST {tet}")
        plt.ylabel('Growth rate')
        plt.legend() 
        plt.show()



# Single tune at starting T
pca_results, pca_tuning, pca_hyperparameters = main(data, tune = TUNE, variation = "pca")
spca_results, spca_tuning, spca_hyperparameters = main(data, tune = TUNE, variation = "spca")
ica_results, ica_tuning, ica_hyperparameters = main(data_ng, tune = TUNE, variation = "ica")

# Tune every T ("TET")
pca_results_TET, pca_hyperparameters_TET = main_tune_every_t(data, tune = TUNE, variation = "pca")
spca_results_TET, spca_hyperparameters_TET = main_tune_every_t(data, tune = TUNE, variation = "spca")
ica_results_TET, ica_hyperparameters_TET = main_tune_every_t(data_ng, tune = TUNE, variation = "ica")


make_individual_plots(pca_results, "XG Boost", "PCA")
make_individual_plots(spca_results, "XG Boost", "SPCA")
make_individual_plots(ica_results, "XG Boost", "ICA")

make_individual_plots(pca_results_TET, "XG Boost", "PCA")
make_individual_plots(spca_results_TET, "XG Boost", "SPCA")
make_individual_plots(ica_results_TET, "XG Boost", "ICA")

joint_results = {"pca" : pca_results, "spca" : spca_results, "ica" : ica_results}
# joint_results = {"pca" : pca_results, "ica" : ica_results}
make_joint_plot(joint_results, "")

joint_results_TET = {"pca" : pca_results_TET, "spca" : spca_results_TET, "ica" : ica_results_TET}
# joint_results_TET = {"pca" : pca_results_TET, "ica" : ica_results_TET}
make_joint_plot(joint_results_TET, "(tet)")



def save_dictionary_to_excel(data, file_name):
    """
    Parameters
    ----------
    data : DICTIONARY (which will save every element on a separate excel-sheet)
            - Key = name DF (sheet)
            - value = DF
    file_name : name of file which is saved
    
    """
    # Saving per weekday data in separate excel sheets
    save_path = directory_export + f"/{file_name}.xlsx"
    writer = pd.ExcelWriter(save_path, engine = 'openpyxl')
    
    # Adding the DataFrames to the excel as a new sheet
    for key, value in data.items():
        value.to_excel(writer, index = False, sheet_name = key) 
        
    writer.save()
    writer.close()



all_results = {
    "PCA"    : pca_results,
    "ICA"    : ica_results,
    "SPCA"   : spca_results,
    "PCA_hyperparameters"    : pca_hyperparameters,
    "ICA_hyperparameters"    : ica_hyperparameters,
    "SPCA_hyperparameters"   : spca_hyperparameters,
    "PCA_TET"    : pca_results_TET,
    "ICA_TET"    : ica_results_TET,
    "SPCA_TET"   : spca_results_TET,
    "PCA_hyperparameters_TET"    : pca_hyperparameters_TET,
    "ICA_hyperparameters_TET"    : ica_hyperparameters_TET,
    "SPCA_hyperparameters_TET"   : spca_hyperparameters_TET
    }

if SAVE == True:
    save_dictionary_to_excel(all_results, file_name = "Hybrid_Boosting_2")



# Print execution time
print(f"total runtime:  {datetime.now() - START_TIME}")
    
        
