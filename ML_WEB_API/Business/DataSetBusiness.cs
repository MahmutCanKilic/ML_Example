using Data;
using DataAccess;
using System;

namespace Business
{
    public class DataSetBusiness
    {
        private readonly DataSetRepository repository;

        public DataSetBusiness(DataSetRepository repository)
        {
            this.repository = repository;
        }

        public DataSetModel GetDatas()
        {
            return repository.GetDatas();
        }
    }
}
