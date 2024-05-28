using Data;
using System;
using System.IO;
using System.Text;

namespace DataAccess
{
    public class DataSetRepository
    {
        private readonly DataSetDbContext db;
        public DataSetRepository(DataSetDbContext db)
        {
            this.db = db;
        }

        public DataSetModel GetDatas()
        {
            var dataBytes = File.ReadAllBytes(@"C:\Users\P2635\ML_Example\breast-cancer.csv");
            string dataString = Encoding.UTF8.GetString(dataBytes);
            var dataModel = new DataSetModel() { Data = dataString};
            db.DataSetModels.Add(dataModel);
            db.SaveChanges();
            return dataModel;
        }
    }
}
