import numpy as np
import xlsxwriter


def write_on_xlsx(train_matrix, test_matrix, filename):
    workbook = xlsxwriter.Workbook(filename)

    worksheet = workbook.add_worksheet(name='Train_results')
    row = 0
    for col, data in enumerate(train_matrix.transpose()):
        worksheet.write_column(row, col, data)


    worksheet = workbook.add_worksheet(name='Test_results')
    row = 0
    for col, data in enumerate(test_matrix.transpose()):
        worksheet.write_column(row, col, data)

    workbook.close()

def extract_results_minibatches(filename, results_dir):
    results_file = open(filename, "r")

    train_results = np.zeros([100, 5], dtype='float32')
    test_results = np.zeros([100, 5], dtype='float32')

    text = results_file.readline()

    row_tr, row_test = 0, 0

    while text:
        text = results_file.readline()
        if "Train measures" in text:
            loss_line = results_file.readline()
            acc_line = results_file.readline().replace("%", "")
            prec_line = results_file.readline().replace("%", "")
            rec_line = results_file.readline().replace("%", "")
            fscore_line = results_file.readline().replace("%", "")
            train_results[row_tr, :] = [float(loss_line.split(': ')[1]), float(acc_line.split(': ')[1]), float(prec_line.split(': ')[1]), float(rec_line.split(': ')[1]), float(fscore_line.split(': ')[1])]
            row_tr = row_tr + 1

        if "Test measures" in text:
            loss_line = results_file.readline()
            acc_line = results_file.readline().replace("%", "")
            prec_line = results_file.readline().replace("%", "")
            rec_line = results_file.readline().replace("%", "")
            fscore_line = results_file.readline().replace("%", "")
            test_results[row_test, :] = [float(loss_line.split(': ')[1]), float(acc_line.split(': ')[1]), float(prec_line.split(': ')[1]), float(rec_line.split(': ')[1]), float(fscore_line.split(': ')[1])]
            row_test = row_test + 1


    write_on_xlsx(train_results, test_results, results_dir)

    results_file.close()

def extract_results_classweight(filename, results_dir, epochs=100):
    results_file = open(filename, "r")

    train_results = np.zeros([epochs, 5], dtype='float32')
    test_results = np.zeros([epochs, 5], dtype='float32')

    text = results_file.readline()

    row_tr, row_test = 0, 0
    index = 0

    while text:
        text = results_file.readline()
        if "val_loss: " in text:
            values = text.split(' - ')
            for i in xrange(len(values)):
                if 'val_loss: ' in values[i]:
                    test_loss = float(values[i].split(': ')[1])
                    index = i
                elif 'val_acc: ' in values[i]:
                    test_acc = float(values[i].split(': ')[1])
                elif 'val_precision: ' in values[i]:
                    test_prec = float(values[i].split(': ')[1])
                elif 'val_recall: ' in values[i]:
                    test_rec = float(values[i].split(': ')[1])
                elif 'val_fmeasure: ' in values[i]:
                    test_fmeas = float(values[i].split(': ')[1])

            train_loss, train_acc, train_prec, train_rec, train_fmeas = float(values[index-5].split(': ')[1]), float(values[index-4].split(': ')[1]), float(values[index-3].split(': ')[1]), float(values[index-2].split(': ')[1]), float(values[index-1].split(': ')[1])

            train_results[row_tr, :] = [train_loss, train_acc, train_prec, train_rec, train_fmeas]
            test_results[row_test, :] = [test_loss, test_acc, test_prec, test_rec, test_fmeas]
            row_tr, row_test = row_tr + 1, row_test + 1

    write_on_xlsx(train_results, test_results, results_dir)

    results_file.close()

if __name__ == "__main__":
    filename = "/home/as6064961/ResultsExtractor/txts_minibatches/out_2_dir_minibatch_lstm32_dp50_epochs100.txt"
    results_dir = "/home/as6064961/ResultsExtractor/Spreadsheets_Minibatches/results_2_dir_lstm32_dp50_epochs100.xlsx"

    extract_results_minibatches(filename, results_dir)
    #extract_results_classweight(filename, results_dir, epochs=100)
