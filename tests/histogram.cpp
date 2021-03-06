

int histogram(const char* Sfile, const char* Outfile, int hpar0 = 10, int hpar1 = 0, int hpar2 = 10){

    TH1D* hist1 = new TH1D("hist1", "hist1", hpar0, hpar1, hpar2);
    hist1->Sumw2();
    TH1D* hist2 = new TH1D("hist2", "hist2", hpar0, hpar1, hpar2);
    hist2->Sumw2();

    int x;
    ifstream inFile;

    inFile.open(Sfile);
    if (!inFile) {
        cout << "Unable to open file";
        exit(1); // terminate with error
    }
    while (inFile >> x) {
        hist1->Fill(x);
        hist2->Fill(x+2);
    }
    inFile.close();

    hist2->Scale(-0.5);
    hist1->Add(hist2);

    hist1->Print("all");

    ofstream myfile;
    myfile.open (Outfile);
    for(int i = 1; i < hpar0 +1; ++i){
        myfile << std::setprecision(9) << hist1->GetBinCenter(i) << "\t" << hist1->GetBinContent(i) << "\t" << hist1->GetBinError(i) << "\n";
    }
    myfile.close();

    gApplication->Terminate();
    return 0;
}